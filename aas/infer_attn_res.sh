for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/dbg/difflmm-attn-v1 \
    --refer-split "refcoco|val" \
    --template "Describe the '{}' in the image." \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/dbg/difflmm-attn-v2 \
    --refer-split "refcoco|val" \
    --template 'Describe the "{}" in the image in detail.' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/dbg/difflmm-attn-v3 \
    --refer-split "refcoco|val" \
    --template 'Describe the "{}" in detail.' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/dbg/difflmm-attn-v4 \
    --refer-split "refcoco|val" \
    --template 'Describe the {} in detail.' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/dbg/difflmm-attn-v5 \
    --refer-split "refcoco|val" \
    --template "Describe the '{}' in detail." \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/dbg/difflmm-attn-v6 \
    --refer-split "refcoco|val" \
    --template 'Where is the "{}"?' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

python aas/vis_attn_ref.py \
    --output-folder save/res/dbg/difflmm-attn-v1 \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --vis-folder save/res/dbg/difflmm-attn-v1-vis \
    --samples 10

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
for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/difflmm-attn-${REFER_SPLITS[$GPU]} \
    --refer-split ${REFER_SPLITS[$GPU]} \
    --template 'Describe the "{}".' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 &
done

for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path liuhaotian/llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/llava-attn-${REFER_SPLITS[$GPU]} \
    --refer-split ${REFER_SPLITS[$GPU]} \
    --template 'Describe the "{}".' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 &
done

REFER_SPLIT="refcoco|val"
for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --ref-anno-folder /mnt/localssd/wglmm/data/refcoco \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --output-folder save/res/difflmm-attn-${REFER_SPLIT} \
    --refer-split ${REFER_SPLIT} \
    --template 'Describe the "{}".' \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done

python aas/vis_attn_ref.py \
    --output-folder save/res/difflmm-attn-${REFER_SPLIT} \
    --image-folder /mnt/localssd/wglmm/data/refcoco/train2014 \
    --vis-folder save/res/difflmm-attn-${REFER_SPLIT}-vis \
    --samples 10
