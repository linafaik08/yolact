"""
Instead of clustering bbox widths and heights, this script
directly optimizes average IoU across the training set given
the specified number of anchor boxes.

Run this script from the Yolact root directory.
"""

import pickle
from itertools import product

import numpy as np
import torch
from scipy.optimize import minimize

dump_file = 'weights/bboxes.pkl'


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b, iscrowd=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    if iscrowd:
        return inter / area_a
    else:
        return inter / union  # [A,B]

# Also convert to point form
def to_relative(bboxes):
	return np.concatenate((bboxes[:, 2:4] / bboxes[:, :2], (bboxes[:, 2:4] + bboxes[:, 4:]) / bboxes[:, :2]), axis=1)


def make_priors(conv_size, scales, aspect_ratios):
	prior_data = []
	conv_h = conv_size[0]
	conv_w = conv_size[1]

	# Iteration order is important (it has to sync up with the convout)
	for j, i in product(range(conv_h), range(conv_w)):
		# +0.5 because priors are in center-size notation
		x = (i + 0.5) / conv_w
		y = (j + 0.5) / conv_h
		
		for scale, ars in zip(scales, aspect_ratios):
			for ar in ars:
				w = scale * ar / conv_w
				h = scale / ar / conv_h

				# Point form
				prior_data += [x, y, x + w, y + h]
	
	return torch.Tensor(prior_data).view(-1, 4).cuda()



scales = [[3.78], [3.72], [3.51], [3.06], [2.70], [2.29]]
aspect_ratios = [[[0.86, 1.51, 0.55]], [[0.84, 1.45, 0.49]], [[0.88, 1.43, 0.52]], [[0.96, 1.61, 0.60]], [[0.91, 1.32, 0.66]], [[0.74, 1.22, 0.90]]]
conv_sizes = [(69, 69), (35, 35), (18, 18), (9, 9), (5, 5), (3, 3)]

batch_idx = 0


def compute_hits(bboxes, anchors, iou_threshold=0.5):
	ious = jaccard(bboxes, anchors)
	perGTAnchorMax, _ = torch.max(ious, dim=1)
	
	return (perGTAnchorMax > iou_threshold)

def compute_recall(hits, base_hits):
	hits = (hits | base_hits).float()
	return torch.sum(hits) / hits.size(0)


def step(x, x_func, bboxes, base_hits, optim_idx):
	# This should set the scale and aspect ratio
	x_func(x, scales[optim_idx], aspect_ratios[optim_idx])

	anchors = make_priors(conv_sizes[optim_idx], scales[optim_idx], aspect_ratios[optim_idx])

	return -float(compute_recall(compute_hits(bboxes, anchors), base_hits).cpu())


def optimize(full_bboxes, optim_idx, batch_size=10000):
	global batch_idx, scales, aspect_ratios, conv_sizes

	start = batch_idx * batch_size
	end   = min((batch_idx + 1) * batch_size, full_bboxes.size(0))

	if batch_idx > (full_bboxes.size(0) // batch_size):
		batch_idx = 0

	bboxes = full_bboxes[start:end, :]

	anchor_base = [
		make_priors(conv_sizes[idx], scales[idx], aspect_ratios[idx])
			for idx in range(len(conv_sizes)) if idx != optim_idx]
	base_hits = compute_hits(bboxes, torch.cat(anchor_base, dim=0))
	
	# def set_x(x, scales, aspect_ratios):
	# 	aspect_ratios[0] = x
		
	def set_x(x, scales, aspect_ratios):
		scales[0] = max(x[0], 0)

	res = minimize(step, x0=scales[optim_idx], method='Powell',
		args = (set_x, bboxes, base_hits, optim_idx),)


def pretty_str(x:list):
	if isinstance(x, list):
		return '[' + ', '.join([pretty_str(y) for y in x]) + ']'
	elif isinstance(x, np.ndarray):
		return pretty_str(list(x))
	else:
		return '%.2f' % x

if __name__ == '__main__':
	
	# Load widths and heights from a dump file. Obtain this with
	# python3 scripts/save_bboxes.py
	with open(dump_file, 'rb') as f:
		bboxes = pickle.load(f)

	# Each box is in the form [im_w, im_h, pos_x, pos_y, size_x, size_y]
	bboxes = np.array(bboxes)
	bboxes = to_relative(bboxes)

	with torch.no_grad():
		bboxes = torch.Tensor(bboxes).cuda()
		
		def print_out():
			print(pretty_str(scales))

		for p in range(10):
			for i in range(len(conv_sizes)):
				print('Sub Iteration %d' % i)
				optimize(bboxes, i)
			
			print('Iteration %d: ' % p, end='')
			print_out()
			print()

			batch_idx += 1


