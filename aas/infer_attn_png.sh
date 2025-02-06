for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_png.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --panoptic-anno data/coco/annotations/panoptic_val2017.json \
    --png-anno data/coco/annotations/png_coco_val2017.json \
    --image-folder data/coco/val2017 \
    --output-folder save/png/difflmm-attn \
    --question "Describe the image in detail." \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_png.py \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --panoptic-anno data/coco/annotations/panoptic_val2017.json \
    --png-anno data/coco/annotations/png_coco_val2017.json \
    --image-folder data/coco/val2017 \
    --output-folder save/png/llava-attn \
    --question "Describe the image in detail." \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done
