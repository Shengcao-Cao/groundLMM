import argparse
import json
import numpy as np
import os
import pycocotools.mask as mask_utils
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-anno', type=str, required=True)
    parser.add_argument('--image-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_anno, 'r') as f:
        anno = json.load(f)

    image_id_to_objs = {}
    for image in anno['images']:
        image_id_to_objs[image['id']] = []
    for obj in anno['annotations']:
        image_id_to_objs[obj['image_id']].append(obj)

    cat_id_to_name = {}
    for cat in anno['categories']:
        cat_id_to_name[cat['id']] = cat['name']

    os.makedirs(args.output_folder, exist_ok=True)
    for image in anno['images'][:100]:
        image_id = image['id']
        image_path = image['file_name']
        image_width = image['width']
        image_height = image['height']
        objs = image_id_to_objs[image_id]
        image_pil = Image.open(os.path.join(args.image_folder, image_path)).convert('RGB')
        for obj in objs:
            obj_id = obj['id']
            cat_id = obj['category_id']
            cat_name = cat_id_to_name[cat_id]
            mask = mask_utils.decode(obj['segmentation'])
            mask_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            mask_map[mask > 0] = 255
            save_image = np.concatenate([np.array(image_pil), mask_map], axis=1)
            save_path = f'{args.output_folder}/{image_id}_{obj_id}_{cat_name}.png'
            Image.fromarray(save_image).save(save_path)
