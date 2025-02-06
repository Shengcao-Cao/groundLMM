python aas/res.py \
    --input-folder /home/shengcaoc/groundLMM/save/res/dbg/difflmm-attn-v1 \
    --segmentation /home/shengcaoc/groundLMM/save/res/co_detr_vit_large_coco_seg_retrain_refcoco_valtest/results_sc0.1_nms0.75.pkl \
    --res-anno /home/shengcaoc/groundLMM/save/res/instances_refcoco_valtest.json

python aas/upper_bound_res.py \
    --input-folder /home/shengcaoc/groundLMM/save/res/dbg/difflmm-attn-v3 \
    --segmentation /home/shengcaoc/groundLMM/save/res/co_detr_vit_large_coco_seg_retrain_refcoco_valtest/results_sc0.1_nms0.75.pkl \
    --res-anno /home/shengcaoc/groundLMM/save/res/instances_refcoco_valtest.json

REFER_SPLITS=(
    "refcoco|val"
    "refcoco|testA"
    "refcoco|testB"
    "refcoco+|val"
    "refcoco+|testA"
    "refcoco+|testB"
    "refcocog|val"
    "refcocog|test"
)
for IDX in 0 1 2 3 4 5 6 7; do
echo ${REFER_SPLITS[$IDX]}
python aas/upper_bound_res.py \
    --input-folder "save/res/llava-attn-${REFER_SPLITS[$IDX]}" \
    --segmentation /home/shengcaoc/groundLMM/save/res/co_detr_vit_large_coco_seg_retrain_refcoco_valtest/results_sc0.1_nms0.75.pkl \
    --res-anno /home/shengcaoc/groundLMM/save/res/instances_refcoco_valtest.json
done
