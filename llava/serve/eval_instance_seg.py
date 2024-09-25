import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--dt', type=str, required=True)
    args = parser.parse_args()

    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
