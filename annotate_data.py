from data import *
from utils.img_to_coco_ann import *

import pandas as pd
import json
from PIL import Image
import os
import argparse


parser = argparse.ArgumentParser(
    description='Annotation Script')

parser.add_argument('--dataset', default=None, type=str,
            help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')

args = parser.parse_args()

if args.dataset is not None:
    set_dataset(args.dataset)


n_images_max = None # limit the number of images to process when debugging

PATH_IMG = cfg.dataset.images
PATH_MASK = cfg.dataset.masks

files_imgs = [os.path.join(PATH_IMG, p) for p in os.listdir(PATH_IMG)]
files_masks = [os.path.join(PATH_MASK, p) for p in os.listdir(PATH_MASK)]

paths_df = pd.DataFrame(files_imgs, columns = ['path_image'])
paths_df['image'] = paths_df.path_image.apply(lambda x: x.split('/')[-1])

paths_m_df = pd.DataFrame(files_masks, columns = ['path_mask'])
paths_m_df['image'] = paths_m_df.path_mask.apply(lambda x: x.split('/')[-1])

paths_df = paths_df.merge(paths_m_df, on='image', how='inner')


category_ids = {cat: i+1 for i,cat in enumerate(CATEGORY_MAP.keys())}
category_colors = {color: i+1 for i,color in enumerate(CATEGORY_MAP.values())}
multipolygon_ids = list(category_ids.values())


coco_format = get_coco_json_format()
coco_format["categories"] = create_category_annotation(category_ids)
coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(
    paths_df, category_colors, multipolygon_ids, n_images_max, threshold_area=0.2)

paths_df.to_csv(os.path.join('data',cfg.dataset.name, 'paths.csv'), index=False)


with open(cfg.dataset.annotation_file, 'w') as f:
    json.dump(coco_format, f)

print('Annotation of data done!')