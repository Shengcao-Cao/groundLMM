import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import decode_segm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--dt', type=str, required=True)
    args = parser.parse_args()

    # COCO evaluation
    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # point accuracy evaluation
    gt = json.load(open(args.gt))
    dt = json.load(open(args.dt))
    image_id_to_image = {}
    image_id_to_gt = {}
    image_id_to_dt = {}
    for image in gt['images']:
        image_id_to_image[image['id']] = image
        image_id_to_gt[image['id']] = []
        image_id_to_dt[image['id']] = []
    for annotation in gt['annotations']:
        image_id_to_gt[annotation['image_id']].append(annotation)
    for annotation in dt:
        image_id_to_dt[annotation['image_id']].append(annotation)

    total = 0
    correct = 0
    for image_id in image_id_to_gt:
        image = image_id_to_image[image_id]
        image_width = image['width']
        image_height = image['height']
        gt_anns = image_id_to_gt[image_id]
        dt_anns = image_id_to_dt[image_id]
        category_id_to_mask_gt = {}
        for annotation in gt_anns:
            category_id = annotation['category_id']
            mask = decode_segm(annotation['segmentation'], image_height, image_width)
            if category_id not in category_id_to_mask_gt:
                category_id_to_mask_gt[category_id] = mask
            else:
                category_id_to_mask_gt[category_id] |= mask
        for annotation in dt_anns:
            category_id = annotation['category_id']
            gt_mask = category_id_to_mask_gt[category_id]
            point_x, point_y = annotation['point']
            if gt_mask[point_y, point_x]:
                correct += 1
            total += 1

    pacc = correct / total * 100.0
    print('Point accuracy: {:.2f}%'.format(pacc))
