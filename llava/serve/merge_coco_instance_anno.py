import argparse
import json
import pycocotools.mask as mask_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-anno', type=str, required=True)
    parser.add_argument('--output-anno', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_anno, 'r') as f:
        anno = json.load(f)

    image_id_to_objs = {}
    for image in anno['images']:
        image_id_to_objs[image['id']] = []
    for obj in anno['annotations']:
        image_id_to_objs[obj['image_id']].append(obj)

    merged_objs = []
    obj_count = 0
    for image in anno['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        objs = image_id_to_objs[image_id]
        cats = set([obj['category_id'] for obj in objs])
        cats = sorted(list(cats))
        for cat in cats:
            cat_objs = [obj for obj in objs if obj['category_id'] == cat]
            mask = None
            for obj in cat_objs:
                if isinstance(obj['segmentation'], list):
                    rles = mask_utils.frPyObjects(obj['segmentation'], image_height, image_width)
                    rle = mask_utils.merge(rles)
                elif isinstance(obj['segmentation']['counts'], list):
                    rle = mask_utils.frPyObjects(obj['segmentation'], image_height, image_width)
                else:
                    rle = obj['segmentation']
                mask_obj = mask_utils.decode(rle)
                if mask is None:
                    mask = mask_obj
                else:
                    mask = mask | mask_obj
            rle = mask_utils.encode(mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            area = int(mask.astype(int).sum())
            obj_count += 1
            merged_objs.append({
                'id': obj_count,
                'image_id': image_id,
                'category_id': cat,
                'segmentation': rle,
                'area': area,
                'bbox': mask_utils.toBbox(rle).tolist(),
                'iscrowd': 0,
            })

    anno['annotations'] = merged_objs
    with open(args.output_anno, 'w') as f:
        json.dump(anno, f, indent=2)
