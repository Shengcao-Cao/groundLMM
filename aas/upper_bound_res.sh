REFER_SPLIT="refcoco|val"
python aas/upper_bound_res.py \
    --input-folder "save/res/dbg/difflmm-attn-v1" \
    --segmentation /home/shengcaoc/groundLMM/save/res/co_detr_vit_large_coco_seg_retrain_refcoco_valtest/results_sc0.1_nms0.75.pkl \
    --res-anno /home/shengcaoc/groundLMM/save/res/instances_refcoco_valtest.json
