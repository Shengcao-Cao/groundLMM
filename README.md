# Emerging Pixel Grounding in Large Multimodal Models Without Grounding Supervision

This is the official PyTorch implementation of our paper:

**Emerging Pixel Grounding in Large Multimodal Models Without Grounding Supervision**

[[Project Page]](https://groundlmm.github.io/) [[Paper]](https://arxiv.org/abs/2410.08209)

[Shengcao Cao](https://shengcao-cao.github.io/), [Liang-Yan Gui](https://lgui.web.illinois.edu/), [Yu-Xiong Wang](https://yxw.web.illinois.edu/)

## 🔎 Overview
![teaser](images/teaser.gif)

We find that the grounding ability can in fact emerge in Large Multimodal Models (LMMs) trained without explicit grounding supervision. To reveal this emerging grounding, we introduce an "attend-and-segment" method which leverages attention maps from standard LMMs to perform pixel-level segmentation. Furthermore, to enhance the grounding ability, we propose DiffLMM, an LMM utilizing a diffusion-based visual encoder, as opposed to the standard CLIP visual encoder, and trained with the same weak supervision. Without being constrained by the biases and limited scale of grounding-specific supervision data, our approach is more generalizable and scalable.

## 🛠️ Installation
Our code is mainly based on [LLaVA](https://github.com/haotian-liu/LLaVA) and the major dependencies are the same. In addition, we need [diffusers](https://github.com/huggingface/diffusers) for DiffLMM, [SAM](https://github.com/facebookresearch/segment-anything) and [spaCy](https://spacy.io) for attend-and-segment.
```
# clone this repo
git clone https://github.com/Shengcao-Cao/groundLMM.git
cd groundLMM

# create conda virtual environment
conda create -n ground-lmm python=3.10 -y
conda activate ground-lmm

# install LLaVA dependencies
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# install diffusion model dependencies
pip install diffusers[torch]==0.15.0

# install SAM dependencies
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# install spaCy dependencies
pip install spacy numpy==1.26.4
python -m spacy download en_core_web_lg
```

If you would like to try our attend-and-segment method on other LMMs, you may directly start from their required environments, and just install SAM and spaCy in addition.

## 🎨 Usage
### 🍉 DiffLMM
#### Minimal Example
DiffLMM can be used just like LLaVA-1.5-7B. We have uploaded our DiffLMM to [here](https://huggingface.co/Shengcao1006/difflmm-llava-v1.5-7b-lora) in HuggingFace. Please note that this checkpoint only includes the LoRA weights, and thus the base model [Vicuna-1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) should always be included when using our model.

For example, you may have a conversation with DiffLMM just like LLaVA:
```
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-file images/llava_logo.png \
    --conv-mode llava_v1 \
    --temperature 0.2 \
    --max-new-tokens 512
```

#### Model Loading
In your code, you may load the model like this:
```
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = 'Shengcao1006/difflmm-llava-v1.5-7b-lora'
model_base = 'lmsys/vicuna-7b-v1.5'

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=model_base,
    model_name=get_model_name_from_path(model_path)
)
```

#### Model Training
If you want to reproduce the training of DiffLMM, please follow the instruction of [LLaVA-1.5](https://github.com/haotian-liu/LLaVA#train), and add/change the following configurations to the [pretraining](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh) and [finetuning](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh) scripts:
```
    --vision_tower stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --mm_projector_type SDCLIPBlock \
    --mm_vision_select_layer 1 \
    --mm_vision_sd_timestep 100 \
    --mm_vision_sd_ensemble_size 1 \
    --mm_vision_sd_clip openai/clip-vit-large-patch14-336 \
    --mm_vision_sd_concat_clip True \
    --mm_vision_sd_implicit_caption True \
    --mm_vision_sd_pe 576 \
    --mm_vision_resolution 384 \
```

### 🍎 Attend-and-Segment

#### Demo
We provide an example to visualize the results from the entire pipeline of attend-and-segment as follows:
```
CUDA_VISIBLE_DEVICES=0 python aas/example.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-path /path/to/image/folder \
    --output-path /path/to/visualization/folder \
    --batch-mode \
    --samples 100 \
    --question "Describe the image in detail." \
    --temperature 0.0 \
    --sam-ckpt checkpoints/sam_vit_h_4b8939.pth
```

If you want to apply the method on one single image, remove `--batch-mode` and provide input/output image paths instead.

#### Extract Attention Maps
In various tasks, we adopt a two-stage processing approach: We first acquire the attention maps during LMM inference, and then use spaCy and SAM to generate pixel-level grounding results. Therefore, we first run `aas/infer_attn.py`:
```
CUDA_VISIBLE_DEVICES=0 python aas/infer_attn.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-folder /path/to/image/folder \
    --output-folder /path/to/attn/folder \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
```

You can use multiple GPUs to accelerate:
```
for GPU in 0 1 2 3 4 5 6 7; do
CUDA_VISIBLE_DEVICES=$GPU python aas/infer_attn.py \
    --model-path Shengcao1006/difflmm-llava-v1.5-7b-lora \
    --model-base lmsys/vicuna-7b-v1.5 \
    --image-folder /path/to/image/folder \
    --output-folder /path/to/attn/folder \
    --temperature 0.0 \
    --feature-height 24 \
    --feature-width 24 \
    --num-chunks 8 \
    --chunk-idx $GPU &
done
```

For newer models, you may revise `aas/infer_attn.py` to generate corresponding attention maps. We provide two examples:
- For [Cambrian-1-8B](https://huggingface.co/nyu-visionx/cambrian-8b): `aas/more_models/cambrian.py`
- For [LLaVA-NeXT-8B](https://huggingface.co/lmms-lab/llama3-llava-next-8b): `aas/more_models/llava_next.py`

You may then use `aas/vis_attn.py` to visualize and verify the generated attention maps.

#### Instance Segmentation
After generating the attention maps based on [COCO](https://cocodataset.org/#download) validation images, we further produce the segmentation results:
```
CUDA_VISIBLE_DEVICES=0 python aas/instance_seg.py \
    --input-folder /path/to/results/instance_seg_attn/difflmm \
    --output-folder /path/to/results/instance_seg/difflmm \
    --ref-anno /path/to/coco/annotations/instances_val2017.json \
    --image-folder /path/to/coco/val2017 \
    --tokenizer lmsys/vicuna-7b-v1.5 \
    --sam-ckpt checkpoints/sam_vit_h_4b8939.pth \
    --more-masks \
    --category-thresh 0.5
```

The results are evaluated as:
```
python aas/eval_instance_seg.py \
    --gt /path/to/coco/annotations/instances_val2017.json \
    --dt /path/to/results/instance_seg/difflmm.json
```

#### Grounded Conversation Generation
Similarly, after generating the attention maps from [GranD-f](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/datasets.md#1-grand-f-grounded-conversation-generation-gcg-dataset) images, we produce the segmentation results:
```
CUDA_VISIBLE_DEVICES=0 python aas/gcg.py \
    --input-folder /path/to/results/gcg_attn/difflmm \
    --output-folder /path/to/results/gcg/difflmm \
    --ref-anno /path/to/GranDf/annotations/val_test/val_gcg_coco_mask_gt.json \
    --image-folder /path/to/GranDf_HA_images/val_test \
    --sam-ckpt checkpoints/sam_vit_h_4b8939.pth \
    --aspect-ratio pad
```

You can also use multiple GPUs to process in parallel by setting `--num-chunks` and `--chunk-idx`. The results are evaluated as below(`pycocoevalcap` is required for this evaluation):
```
CUDA_VISIBLE_DEVICES=0 python aas/eval_gcg.py \
    --pd-folder /path/to/results/gcg/difflmm \
    --gt-caption /path/to/GranDf/annotations/val_test/val_gcg_coco_caption_gt.json \
    --gt-mask /path/to/GranDf/annotations/val_test/val_gcg_coco_mask_gt.json \
    --split val
```

## 🙏 Acknowledgements
Our work is greatly inspired by the following repositories:
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [DIFT](https://github.com/Tsingularity/dift)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [GLaMM](https://github.com/mbzuai-oryx/groundingLMM)

We greatly appreciate their open-source work!

## ⚖️ License
This project is released under the Apache 2.0 license. Other codes from open source repository follows the original distributive licenses.

## 🌟 Citation
If you find our research interesting or use our code, model, or method in your research, please consider citing our work.
```
@article{cao2024emerging,
  title={Emerging Pixel Grounding in Large Multimodal Models Without Grounding Supervision},
  author={Cao, Shengcao and Gui, Liang-Yan and Wang, Yu-Xiong},
  journal={arXiv preprint arXiv:2410.08209},
  year={2024}
}
```