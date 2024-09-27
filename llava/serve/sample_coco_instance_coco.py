import argparse
import json
import numpy as np
import pycocotools.mask as mask_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-anno', type=str, required=True)
    parser.add_argument('--output-anno', type=str, required=True)
    parser.add_argument('--samples', type=int, default=1000)
    args = parser.parse_args()

    with open(args.input_anno, 'r') as f:
        anno = json.load(f)

    image_id_to_objs = {}
    for image in anno['images']:
        image_id_to_objs[image['id']] = []
    for obj in anno['annotations']:
        image_id_to_objs[obj['image_id']].append(obj)

    sampled_images = np.random.choice(anno['images'], args.samples, replace=False)
    sampled_images = sorted(sampled_images, key=lambda x: x['id'])
    sampled_objs = []
    for image in sampled_images:
        image_id = image['id']
        objs = image_id_to_objs[image_id]
        sampled_objs.extend(objs)

    anno['images'] = sampled_images
    anno['annotations'] = sampled_objs
    with open(args.output_anno, 'w') as f:
        json.dump(anno, f, indent=2)
