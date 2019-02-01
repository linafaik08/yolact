# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, center_size
from utils.functions import sanitize_coordinates

from data import cfg, mask_type, activation_func

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        
        # Extra loss coefficients to get all the losses to be in a similar range
        self.mask_alpha = 0.4 / 256 * 140 * 140 # We'll divide this by mask_h and mask_w later
        self.bbox_alpha = 5 if cfg.use_yolo_regressors else 1

        if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
            self.mask_alpha *= 30
        if cfg.mask_proto_reweight_mask_loss:
            self.mask_alpha /= 4
        if cfg.mask_proto_crop and cfg.mask_proto_normalize_emulate_roi_pooling:
            self.mask_alpha *= 0.2

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20*20/70/70
        self.l1_alpha = 0.1

    def forward(self, predictions, wrapper, wrapper_mask):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
            
            * Only if mask_type == lincomb
        """

        loc_data  = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        priors    = predictions['priors']

        if cfg.mask_type == mask_type.lincomb:
            proto_data = predictions['proto']
        
        if cfg.use_instance_coeff:
            inst_data = predictions['inst']
        else:
            inst_data = None
        
        targets, masks, num_crowds = wrapper.get_args(wrapper_mask)

        num = loc_data.size(0)
        # This is necessary for training on multiple GPUs because
        # DataParallel will cat the priors from each GPU together
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(num, num_priors, 4)
        gt_box_t = loc_data.new(num, num_priors, 4)
        conf_t = loc_data.new(num, num_priors).long()
        idx_t = loc_data.new(num, num_priors).long()

        defaults = priors.data

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data

            # Split the crowd annotations because they come bundled in
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths = split(truths)

                # We don't use the crowd labels or masks
                _, labels = split(labels)
                _, masks[idx] = split(masks[idx])
            else:
                crowd_boxes = None

            
            match(self.pos_threshold, self.neg_threshold,
                  truths, defaults, labels, crowd_boxes,
                  loc_t, conf_t, idx_t, idx, loc_data[idx])
                  
            gt_box_t[idx, :, :] = truths[idx_t[idx]]

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        
        # Localization Loss (Smooth L1)
        loss_l = 0
        if cfg.train_boxes:
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') * self.bbox_alpha

        loss_m = 0 # Mask Loss
        loss_p = 0 # Prototype Loss
        if cfg.train_masks:
            if cfg.mask_type == mask_type.direct:
                if cfg.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(num):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = torch.cat(pos_masks, 0)
                    masks_p = mask_data[pos, :].view(-1, cfg.mask_dim)
                    loss_m = F.binary_cross_entropy(masks_p, masks_t, reduction='sum') * self.mask_alpha
                else:
                    loss_m = self.direct_mask_loss(pos_idx, idx_t, loc_data, mask_data, priors, masks)
            elif cfg.mask_type == mask_type.lincomb:
                loss_m = self.lincomb_mask_loss(pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, inst_data)
                
                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        loss_p = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        loss_p = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])

        # Confidence loss
        loss_c = 0
        if cfg.use_focal_loss:
            loss_c = self.focal_conf_loss(conf_data, conf_t)
        else:
            loss_c = self.ohem_conf_loss(conf_data, conf_t, pos, num)

        # Divide all losses by the number of positives.
        # Don't do it for loss_p because that doesn't depend on the anchors.
        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        loss_m /= N
        return loss_l, loss_c, loss_m + loss_p

    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        if cfg.ohem_use_most_confident:
            # i.e. max(softmax) along classes > 0 
            batch_conf = F.softmax(batch_conf, dim=1)
            loss_c, _ = batch_conf[:, 1:].max(dim=1)
        else:
            # i.e. -softmax(class 0 confidence)
            loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]
        
        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos]        = 0 # filter out pos boxes
        loss_c[conf_t < 0] = 0 # filter out neutrals (conf_t = -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos]        = 0
        neg[conf_t < 0] = 0 # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        
        return loss_c

    def focal_conf_loss(self, conf_data, conf_t, loss_alpha=0.05):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        Adapted from https://github.com/pytorch/pytorch/blob/master/modules/detectron/sigmoid_focal_loss_op.cu
        """

        # Ignore neutral samples
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0 # so that scatter doesn't drum up a fuss

        # Construct a 1-hot encoding of the true classes
        one_hot_t = conf_data.new_full(conf_data.size(), 0, requires_grad=False).view(-1, conf_data.size(-1))
        one_hot_t.scatter_(1, conf_t.view(-1, 1), 1)
        one_hot_t = one_hot_t.view(conf_data.size())

        # Note this uses sigmoid, but we use softmax at test time. Could be a problem? idk
        pred = torch.sigmoid(conf_data)

        term1 = torch.pow(1 - pred, cfg.focal_loss_gamma) * torch.log(pred + 0.00001)
        term2 = torch.pow(pred, cfg.focal_loss_gamma) * (-1 * conf_data * (conf_data >= 0).float() -
                                                         torch.log(1 + torch.exp(conf_data - 2 * conf_data * (conf_data >= 0).float())))

        loss1 = (-one_hot_t * term1 * cfg.focal_loss_alpha)
        loss2 = (-(1-one_hot_t) * term2 * (1 - cfg.focal_loss_alpha))

        return loss_alpha * (keep * (loss1 + loss2).sum(-1)).sum()

    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]

                # Shape: [num_priors, 4], decoded predicted bboxes
                pos_bboxes = decode(loc_data[idx, :, :], priors.data)
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]
                
                # Convert bboxes to absolute coordinates
                num_pos, img_height, img_width = pos_masks.size()

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)

                # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
                # Note that each bounding box crop is a different size so I don't think we can vectorize this
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]

                    # Restore any dimensions we've left out because our bbox was 1px wide
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)

                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                    scaled_masks.append(new_mask.view(1, -1))

                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float() # Threshold downsampled mask
            
            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += F.binary_cross_entropy(pos_mask_data, mask_t, reduction='sum') * self.mask_alpha

        return loss_m
    

    def lincomb_mask_loss(self, pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, inst_data, interpolation_mode='bilinear'):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        process_gt_bboxes = cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop

        if cfg.mask_proto_remove_empty_masks:
            # Make sure to store a copy of this because we edit it to get rid of all-zero masks
            pos = pos.clone()

        loss_m = 0
        loss_c = 0 # Coefficient loss

        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()

                if cfg.mask_proto_remove_empty_masks:
                    # Get rid of gt masks that are so small they get downsampled away
                    very_small_masks = (downsampled_masks.sum(dim=(0,1)) <= 0.0001)
                    for i in range(very_small_masks.size(0)):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0

                if cfg.mask_proto_reweight_mask_loss:
                    # Ensure that the gt is binary
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = downsampled_masks.gt(0.5).float()
                    else:
                        bin_gt = downsampled_masks

                    gt_foreground_norm = bin_gt     / (torch.sum(bin_gt,   dim=(0,1), keepdim=True) + 0.0001)
                    gt_background_norm = (1-bin_gt) / (torch.sum(1-bin_gt, dim=(0,1), keepdim=True) + 0.0001)

                    mask_reweighting   = gt_foreground_norm * cfg.mask_proto_reweight_coeff + gt_background_norm
                    mask_reweighting  *= mask_h * mask_w

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            
            if process_gt_bboxes:
                # Note: this is in point-form
                pos_gt_box_t = gt_box_t[idx, cur_pos]

            if pos_idx_t.size(0) == 0:
                continue

            proto_masks = proto_data[idx]
            proto_coef  = mask_data[idx, cur_pos, :]

            if cfg.mask_proto_coeff_diversity_loss:
                if inst_data is not None:
                    div_coeffs = inst_data[idx, cur_pos, :]
                else:
                    div_coeffs = proto_coef

                norm_coeff = F.normalize(div_coeffs, dim=1)
                select = torch.randperm(norm_coeff.size(0))

                # Note that I don't account for fixed points
                perm_idx_t = pos_idx_t[select]
                perm_coeff = norm_coeff[select, :]

                cos_sim = (torch.sum(norm_coeff * perm_coeff, dim=1) + 1) / 2

                # If they're the same instance, use coefficient distance, else use coefficient similarity
                same_instance = (pos_idx_t == perm_idx_t).float()
                loss = (1 - cos_sim) * same_instance + cos_sim * (1 - same_instance)

                loss_c += torch.sum(loss) * 2
            
            # If we have over the allowed number of masks, select a random sample
            if proto_coef.size(0) > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:cfg.masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_idx_t  = pos_idx_t[select]
                
                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[select, :]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]          

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.t()
            pred_masks = cfg.mask_proto_mask_activation(pred_masks)

            if cfg.mask_proto_crop:
                # Shortening the variable name here
                expnd = cfg.mask_proto_crop_expand

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                # Note I tried to put expand here but the loss exploded for some reason
                x1, x2 = sanitize_coordinates(pos_gt_box_t[:, 0], pos_gt_box_t[:, 2], mask_w)
                y1, y2 = sanitize_coordinates(pos_gt_box_t[:, 1], pos_gt_box_t[:, 3], mask_h)

                # "Crop" predicted masks by zeroing out everything not in the predicted bbox
                # TODO: Write a cuda implementation of this to get rid of the loop
                crop_mask = torch.zeros(mask_h, mask_w, num_pos)
                for jdx in range(num_pos):
                    crop_mask[y1[jdx]*(1-expnd):y2[jdx]*(1+expnd), x1[jdx]*(1-expnd):x2[jdx]*(1+expnd), jdx] = 1
                pred_masks = pred_masks * crop_mask
            
            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')

            if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
                gt_area  = torch.sum(mask_t, dim=(0, 1), keepdim=True)
                pre_loss = pre_loss / (torch.sqrt(gt_area) + 0.0001)
            
            if cfg.mask_proto_reweight_mask_loss:
                pre_loss = pre_loss * mask_reweighting[:, :, pos_idx_t]
                
            if cfg.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if cfg.mask_proto_crop else 1
                pos_get_csize = center_size(pos_gt_box_t)
                gt_box_width  = pos_get_csize[:, 2]
                gt_box_height = pos_get_csize[:, 3]
                pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight


            loss_m += torch.sum(pre_loss)

        return loss_m * self.mask_alpha / mask_h / mask_w + loss_c
