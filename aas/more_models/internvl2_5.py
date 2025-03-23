import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from internvl2_5_image_loader import load_image

import math


def split_list(lst, n):
    '''Split a list into n (roughly) equal-sized chunks'''
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        use_flash_attn=False,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id
    eos_token_id = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())

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

        pixel_values, patch_dim = load_image(image_path, min_num=args.min_image_crops, max_num=args.max_image_crops)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        num_patches_list = [pixel_values.shape[0]]

        qs = '<image>\n' + args.question
        conv = model.conv_template.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = '<img>' + '<IMG_CONTEXT>' * model.num_image_token * num_patches + '</img>'
            prompt = prompt.replace('<image>', image_tokens, 1)

        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

        image_token_indices = inputs.input_ids[0].eq(img_context_token_id).nonzero(as_tuple=False)
        image_token_start = int(image_token_indices[0])
        image_token_end = int(image_token_indices[-1]) + 1

        with torch.inference_mode():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                eos_token_id=eos_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        save_sequences = output_ids['sequences'][0].detach().cpu()
        answer = tokenizer.decode(save_sequences, skip_special_tokens=True).strip()
        save_attn = []
        for i in range(len(output_ids['attentions'])):
            save_attn_i = output_ids['attentions'][i]
            save_attn_i = torch.cat([x[:, :, -1, image_token_start:image_token_end] for x in save_attn_i])
            save_attn_i = save_attn_i.mean(dim=(0, 1))
            patches_per_side = int(model.num_image_token ** 0.5)
            if args.use_thumbnail_attention:
                assert patch_dim[0] == 1
                save_attn_i = save_attn_i[-model.num_image_token:]
                save_attn_i = save_attn_i.reshape(patches_per_side, patches_per_side)
            else:
                assert patch_dim[0] in [0, 1]
                if patch_dim[0] == 1 and patch_dim[1] * patch_dim[2] > 1:
                    save_attn_i = save_attn_i[:-model.num_image_token]
                save_attn_i = save_attn_i.reshape(patch_dim[2], patch_dim[1], patches_per_side, patches_per_side)
                save_attn_i = save_attn_i.permute(0, 2, 1, 3).contiguous()
                save_attn_i = save_attn_i.reshape(patch_dim[2] * patches_per_side, patch_dim[1] * patches_per_side)
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
    parser.add_argument('--model-path', type=str, default='OpenGVLab/InternVL2_5-8B')
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
    parser.add_argument('--min-image-crops', type=int, default=3)
    parser.add_argument('--max-image-crops', type=int, default=9)
    parser.add_argument('--use-thumbnail-attention', action='store_true')
    args = parser.parse_args()

    eval_model(args)
