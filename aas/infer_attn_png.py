import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

from utils import split_list, get_chunk


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.get_vision_tower().to(dtype=torch.float16)

    pan_anno = json.load(open(args.panoptic_anno))
    image_id_to_image = {}
    for image in pan_anno['images']:
        image_id_to_image[image['id']] = image

    png_anno = json.load(open(args.png_anno))
    anno_indices = get_chunk(list(range(len(png_anno))), args.num_chunks, args.chunk_idx)
    os.makedirs(args.output_folder, exist_ok=True)

    for anno_index in tqdm(anno_indices):
        image_id = int(png_anno[anno_index]['image_id'])
        image = image_id_to_image[image_id]
        image_file = image['file_name']
        image_path = os.path.join(args.image_folder, image_file)
        attn_path = os.path.join(args.output_folder, f'{anno_index:012d}.pth')
        qs = args.question
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        caption_tokens = []
        token_groups = []
        for segments in png_anno[anno_index]['segments']:
            utterance = segments['utterance']
            tokens = tokenizer.encode(utterance, add_special_tokens=False)
            token_groups.append(list(range(len(caption_tokens), len(caption_tokens) + len(tokens))))
            caption_tokens.extend(tokens)

        caption_token_length = len(caption_tokens)
        caption = tokenizer.decode(caption_tokens)

        input_ids = torch.cat([input_ids, torch.tensor(caption_tokens, dtype=input_ids.dtype).unsqueeze(0).cuda()], dim=1)

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True)

            # automatically detect image tokens
            image_token_start_index = -1
            for i in range(input_ids.shape[1]):
                if input_ids[0, i] == IMAGE_TOKEN_INDEX:
                    image_token_start_index = i
                    break
            assert image_token_start_index >= 0

            # process and save attention
            # save_sequences = output_ids['sequences'][0].detach().cpu()
            save_sequences = input_ids[0, -caption_token_length:].detach().cpu()
            save_attn = torch.cat(output_ids['attentions'][0])
            image_token_length = save_attn.shape[-1] - input_ids.shape[1] + 1
            if args.reg_tokens > 0:
                image_token_length -= args.reg_tokens
                image_token_start_index += args.reg_tokens
            image_token_end_index = image_token_start_index + image_token_length
            assert image_token_length == args.feature_height * args.feature_width, \
                f'Image token length mismatch: Expected {args.feature_height * args.feature_width}, got {image_token_length}'
            save_attn = save_attn[:, :, -caption_token_length:, image_token_start_index:image_token_end_index]
            save_attn = save_attn.mean(dim=(0, 1))
            save_attn = save_attn.reshape(caption_token_length, args.feature_height, args.feature_width)
            save_attn = save_attn.detach().cpu()

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
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--panoptic-anno', type=str, default='')
    parser.add_argument('--png-anno', type=str, default='')
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--question', type=str, default='Describe the image in detail.')
    parser.add_argument('--conv-mode', type=str, default='llava_v1')
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--sample', type=str, default=None)
    parser.add_argument('--feature-height', type=int, default=24)
    parser.add_argument('--feature-width', type=int, default=24)
    parser.add_argument('--reg-tokens', type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
