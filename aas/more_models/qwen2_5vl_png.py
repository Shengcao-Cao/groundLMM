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

    pan_anno = json.load(open(args.panoptic_anno))
    image_id_to_image = {}
    for image in pan_anno['images']:
        image_id_to_image[image['id']] = image

    png_anno = json.load(open(args.png_anno))
    anno_indices = list(range(len(png_anno)))
    if args.resume:
        existing_attn_files = os.listdir(args.output_folder)
        existing_attn_files = set([x.split('.')[0] for x in existing_attn_files])
        anno_indices = [x for x in anno_indices if f'{x:012d}' not in existing_attn_files]
        print(f'Resuming from {len(existing_attn_files)} files, {len(anno_indices)} remaining.')

    anno_indices = get_chunk(anno_indices, args.num_chunks, args.chunk_idx)
    os.makedirs(args.output_folder, exist_ok=True)

    for anno_index in tqdm(anno_indices):
        image_id = int(png_anno[anno_index]['image_id'])
        image = image_id_to_image[image_id]
        image_file = image['file_name']
        image_path = os.path.join(args.image_folder, image_file)
        attn_path = os.path.join(args.output_folder, f'{anno_index:012d}.pth')

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

        caption_tokens = []
        token_groups = []
        for segments in png_anno[anno_index]['segments']:
            utterance = segments['utterance'] + ' '
            tokens = tokenizer.encode(utterance, add_special_tokens=False)
            token_groups.append(list(range(len(caption_tokens), len(caption_tokens) + len(tokens))))
            caption_tokens.extend(tokens)

        caption_token_length = len(caption_tokens)
        caption = tokenizer.decode(caption_tokens)

        inputs.data['input_ids'] = torch.cat([inputs.input_ids, torch.tensor(caption_tokens, dtype=inputs.input_ids.dtype).unsqueeze(0).cuda()], dim=1)
        inputs.data['attention_mask'] = torch.cat([inputs.attention_mask, torch.ones(1, len(caption_tokens), dtype=inputs.attention_mask.dtype).cuda()], dim=1)

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

        save_sequences = inputs.input_ids[0, -caption_token_length:].detach().cpu()
        save_attn = output_ids['attentions'][0]
        save_attn = torch.cat([x[:, :, -caption_token_length:, image_token_start:image_token_end] for x in save_attn])
        save_attn = save_attn.mean(dim=(0, 1)).reshape(caption_token_length, feature_height, feature_width)
        save_attn = save_attn.detach().float().cpu()

        save_dict = {
            'answer': caption,
            'sequences': save_sequences,
            'attentions': save_attn,
            'token_groups': token_groups,
            'image_id': image_id,
            'image_file': image_file,
        }
        torch.save(save_dict, attn_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--panoptic-anno', type=str, default='')
    parser.add_argument('--png-anno', type=str, default='')
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
