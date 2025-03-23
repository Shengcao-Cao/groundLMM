import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import math


def split_list(lst, n):
    '''Split a list into n (roughly) equal-sized chunks'''
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype='auto',
        device_map='auto',
        attn_implementation='eager',
    )
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=1008*1008)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    image_folder = os.path.expanduser(args.image_folder)
    image_files = sorted(os.listdir(image_folder))
    if args.sample is not None:
        image_ids = json.load(open(args.sample))
        image_ids = set(image_ids)
        image_files = [x for x in image_files if x.split('.')[0] in image_ids]

    if args.resume:
        existing_attn_files = os.listdir(args.output_folder)
        existing_attn_files = set([x.split('.')[0] for x in existing_attn_files])
        image_files = [x for x in image_files if x.split('.')[0] not in existing_attn_files]
        print(f'Resuming from {len(existing_attn_files)} files, {len(image_files)} remaining.')

    image_files = get_chunk(image_files, args.num_chunks, args.chunk_idx)
    os.makedirs(args.output_folder, exist_ok=True)

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        attn_path = os.path.join(args.output_folder, image_file.split('.')[0] + '.pth')

        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'image': image_path,
                    },
                    {'type': 'text', 'text': args.question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to('cuda')

        image_grid_thw = inputs.image_grid_thw[0]
        feature_height = int(image_grid_thw[1]) // 2
        feature_width = int(image_grid_thw[2]) // 2
        image_token_indices = inputs.input_ids[0].eq(model.config.image_token_id).nonzero(as_tuple=False)
        image_token_start = int(image_token_indices[0])
        image_token_end = int(image_token_indices[-1]) + 1

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        save_sequences = output_ids['sequences'][0][inputs.input_ids.shape[1]:].detach().cpu()
        answer = tokenizer.decode(save_sequences, skip_special_tokens=True).strip()
        save_attn = []
        for i in range(len(output_ids['attentions'])):
            save_attn_i = output_ids['attentions'][i]
            save_attn_i = torch.cat([x[:, :, -1, image_token_start:image_token_end] for x in save_attn_i])
            save_attn_i = save_attn_i.mean(dim=(0, 1)).reshape(feature_height, feature_width)
            save_attn.append(save_attn_i.detach().float().cpu())

        save_attn = torch.stack(save_attn)
        save_dict = {
            'answer': answer,
            'sequences': save_sequences,
            'attentions': save_attn,
            'image_file': image_file,
        }
        torch.save(save_dict, attn_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--question', type=str, default='Describe the image in detail.')
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--sample', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    eval_model(args)
