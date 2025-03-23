MODEL_NAME="difflmm"
MODEL_PATH="Shengcao1006/difflmm-llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
SAVE_DATA_PATH="save"

# infer attention for all images
for GPU in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --image-folder "data/GranDf_HA_images/val_test" \
    --output-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME-attn" \
    --temperature 0.0 \
    --max_new_tokens 512 \
    --num-chunks 4 \
    --chunk-idx $GPU 2>&1 &
done
wait

# produce segmentation masks for validation set
for GPU in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$GPU python aas/gcg.py \
    --input-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME-attn" \
    --output-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME" \
    --ref-anno "data/GranDf/annotations/val_test/val_gcg_coco_mask_gt.json" \
    --image-folder "data/GranDf_HA_images/val_test" \
    --tokenizer "lmsys/vicuna-7b-v1.5" \
    --sam-ckpt "checkpoints/sam_vit_h_4b8939.pth" \
    --offset 1 \
    --aspect-ratio pad \
    --group-aggregation max \
    --num-chunks 4 \
    --chunk-idx $GPU 2>&1 &
done
wait

# produce segmentation masks for test set
for GPU in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$GPU python aas/gcg.py \
    --input-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME-attn" \
    --output-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME" \
    --ref-anno "data/GranDf/annotations/val_test/test_gcg_coco_mask_gt.json" \
    --image-folder "data/GranDf_HA_images/val_test" \
    --tokenizer "lmsys/vicuna-7b-v1.5" \
    --sam-ckpt "checkpoints/sam_vit_h_4b8939.pth" \
    --offset 1 \
    --aspect-ratio pad \
    --group-aggregation max \
    --num-chunks 4 \
    --chunk-idx $GPU 2>&1 &
done
wait

# evaluate results for validation set
CUDA_VISIBLE_DEVICES=0 python aas/eval_gcg.py \
    --pd-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME" \
    --gt-caption "data/GranDf/annotations/val_test/val_gcg_coco_caption_gt.json" \
    --gt-mask "data/GranDf/annotations/val_test/val_gcg_coco_mask_gt.json" \
    --split val

# evaluate results for test set
CUDA_VISIBLE_DEVICES=0 python aas/eval_gcg.py \
    --pd-folder "$SAVE_DATA_PATH/gcg/$MODEL_NAME" \
    --gt-caption "data/GranDf/annotations/val_test/test_gcg_coco_caption_gt.json" \
    --gt-mask "data/GranDf/annotations/val_test/test_gcg_coco_mask_gt.json" \
    --split test
