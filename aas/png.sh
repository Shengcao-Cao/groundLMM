python aas/png.py \
    --input-folder save/png/difflmm-attn \
    --output-json save/png/difflmm.json \
    --panoptic-anno data/coco/annotations/panoptic_val2017.json \
    --png-anno data/coco/annotations/png_coco_val2017.json \
    --panoptic-pred-folder /home/shengcaoc/OpenSeeD/save/openseed_swinl/panoptic_eval/ \
    --image-folder data/coco/val2017

python aas/png.py \
    --input-folder save/png/llava-attn \
    --output-json save/png/llava.json \
    --panoptic-anno data/coco/annotations/panoptic_val2017.json \
    --png-anno data/coco/annotations/png_coco_val2017.json \
    --panoptic-pred-folder /home/shengcaoc/OpenSeeD/save/openseed_swinl/panoptic_eval/ \
    --image-folder data/coco/val2017
