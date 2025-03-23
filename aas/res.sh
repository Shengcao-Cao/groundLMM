MODEL_NAME="difflmm"
MODEL_PATH="Shengcao1006/difflmm-llava-v1.5-7b-lora"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
SAVE_DATA_PATH="save"

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

# infer attention
for REFER_SPLIT in ${REFER_SPLITS[@]}; do
for GPU in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn_res.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --image-folder "data/coco_2014/train2014" \
    --output-folder "${SAVE_DATA_PATH}/res/${MODEL_NAME}-attn/${REFER_SPLIT}" \
    --ref-anno-folder "data/Refer_Segm" \
    --refer-split ${REFER_SPLIT} \
    --template 'Describe the "{}".' \
    --temperature 0.0 \
    --max_new_tokens 64 \
    --num-chunks 4 \
    --chunk-idx $GPU &
done
wait
done
wait

# produce segmentation results and evaluate
for REFER_SPLIT in ${REFER_SPLITS[@]}; do
echo $REFER_SPLIT
python aas/res.py \
    --input-folder "${SAVE_DATA_PATH}/res/${MODEL_NAME}-attn/${REFER_SPLIT}" \
    --segmentation "data/res/co_detr_retrain_inference_results.pkl" \
    --res-anno "data/res/instances_refcoco_valtest.json" \
    --tokenizer "lmsys/vicuna-7b-v1.5" \
    --offset 1 \
    --aspect-ratio pad
done
wait
