import argparse
import copy
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, select_best_resolution, resize_and_pad_image

from PIL import Image
import math


def split_list(lst, n):
    '''Split a list into n (roughly) equal-sized chunks'''
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_unpad_feature_size(feature_width, feature_height, image_width, image_height):
    image_aspect_ratio = image_width / image_height
    feature_aspect_ratio = feature_width / feature_height
    if image_aspect_ratio > feature_aspect_ratio:
        scale_factor = feature_width / image_width
        new_feature_width = feature_width
        new_feature_height = int(image_height * scale_factor)
        height_padding = (feature_height - new_feature_height) // 2
        width_padding = 0
        new_feature_height = feature_height - height_padding * 2
    else:
        scale_factor = feature_height / image_height
        new_feature_width = int(image_width * scale_factor)
        new_feature_height = feature_height
        width_padding = (feature_width - new_feature_width) // 2
        height_padding = 0
        new_feature_width = feature_width - width_padding * 2
    return new_feature_width, new_feature_height, width_padding, height_padding


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation='eager')

    image_folder = os.path.expanduser(args.image_folder)
    image_files = sorted(os.listdir(image_folder))
    if args.sample is not None:
        image_ids = json.load(open(args.sample))
        image_ids = set(image_ids)
        image_files = [x for x in image_files if x.split('.')[0] in image_ids]

    image_files = get_chunk(image_files, args.num_chunks, args.chunk_idx)
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    os.makedirs(args.output_folder, exist_ok=True)
    attn_paths = [os.path.join(args.output_folder, f.split('.')[0] + '.pth') for f in image_files]

    for i in tqdm(range(len(image_files))):
        image_path = image_paths[i]
        attn_path = attn_paths[i]
        qs = args.question
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        best_resolution = select_best_resolution(image.size, model.config.image_grid_pinpoints)
        image_padded = resize_and_pad_image(image, best_resolution)
        num_patches_per_side = model.get_vision_tower().num_patches_per_side
        patch_size = model.get_vision_tower().config.patch_size
        num_base_patches = num_patches_per_side ** 2
        original_feature_width = best_resolution[0] // patch_size
        original_feature_height = best_resolution[1] // patch_size
        new_feature_width, new_feature_height, width_padding, height_padding = get_unpad_feature_size(original_feature_width, original_feature_height, image.width, image.height)

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
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id)

            # get answer
            answer = tokenizer.decode(output_ids['sequences'][0], skip_special_tokens=True).strip()

            # automatically detect image tokens
            image_token_start_index = -1
            for i in range(input_ids.shape[1]):
                if input_ids[0, i] == IMAGE_TOKEN_INDEX:
                    image_token_start_index = i
                    break
            assert image_token_start_index >= 0

            # process and save attention
            save_sequences = output_ids['sequences'][0].detach().cpu()
            save_attn = []
            for i in range(len(output_ids['attentions'])):
                save_attn_i = output_ids['attentions'][i]                                           # n_layers x n_heads x n_output x n_input
                save_attn_i = torch.cat([x[:, -2:-1, :] for x in save_attn_i])                      # n_layers x n_heads x 1 x n_input
                if i == 0:
                    if args.use_base_patches:
                        image_token_length = num_base_patches
                        image_token_end_index = image_token_start_index + image_token_length
                        new_feature_width = num_patches_per_side
                        new_feature_height = num_patches_per_side
                    else:
                        image_token_length = save_attn_i.shape[-1] - input_ids.shape[1] + 1
                        image_token_end_index = image_token_start_index + image_token_length
                        image_token_start_index += num_base_patches
                        image_token_length -= num_base_patches
                        assert image_token_length == new_feature_height * (new_feature_width + 1), \
                            f'Image token length mismatch: Expected {new_feature_height * (new_feature_width + 1)}, got {image_token_length}'
                save_attn_i = save_attn_i[:, :, -1, image_token_start_index:image_token_end_index]          # n_layers x n_heads x n_image_tokens
                save_attn_i = save_attn_i.mean(dim=(0, 1))                                                  # n_image_tokens
                if args.use_base_patches:
                    save_attn_i = save_attn_i.reshape(new_feature_height, new_feature_width)
                else:
                    save_attn_i = save_attn_i.reshape(new_feature_height, new_feature_width + 1)[:, :-1]    # feature_height x feature_width (excluding the newline token)
                save_attn.append(save_attn_i.detach().cpu())

        save_attn = torch.stack(save_attn)
        save_dict = {
            'answer': answer,
            'sequences': save_sequences,
            'attentions': save_attn,
        }
        torch.save(save_dict, attn_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
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
    parser.add_argument('--use-base-patches', action='store_true')
    args = parser.parse_args()

    eval_model(args)
