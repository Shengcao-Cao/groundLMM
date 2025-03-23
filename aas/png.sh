MODEL_NAME="difflmm"
MODEL_PATH="Shengcao1006/difflmm-llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
SAVE_DATA_PATH="save"

# infer attention
for GPU in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_png.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --image-folder "data/coco/val2017" \
    --output-folder "$SAVE_DATA_PATH/png/$MODEL_NAME-attn" \
    --panoptic-anno "data/coco/annotations/panoptic_val2017.json" \
    --png-anno "data/coco/annotations/png_coco_val2017.json" \
    --temperature 0.0 \
    --num-chunks 4 \
    --chunk-idx $GPU \
    --max_new_tokens 1 2>&1 &
done
wait

# produce segmentation results and evaluate
python aas/png.py \
    --input-folder "$SAVE_DATA_PATH/png/$MODEL_NAME-attn" \
    --output-json "$SAVE_DATA_PATH/png/$MODEL_NAME.json" \
    --panoptic-anno "data/coco/annotations/panoptic_val2017.json" \
    --png-anno "data/coco/annotations/png_coco_val2017.json" \
    --panoptic-pred-folder "data/png/openseed_inference_results" \
    --image-folder "data/coco/val2017" \
    --aspect-ratio pad \
    --group-aggregation max
