import argparse
import torch
import os
import json
import pickle
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image

from utils import get_chunk, decode_segm


def process_ref_anno(args):
    refer_dataset, refer_split = args.refer_split.split('|')
    if refer_dataset == 'refcoco':
        ref_annos_file = os.path.join(args.ref_anno_folder, 'refcoco', 'refs(unc).p')
        ref_instances_file = os.path.join(args.ref_anno_folder, 'refcoco', 'instances.json')
    elif refer_dataset == 'refcoco+':
        ref_annos_file = os.path.join(args.ref_anno_folder, 'refcoco+', 'refs(unc).p')
        ref_instances_file = os.path.join(args.ref_anno_folder, 'refcoco+', 'instances.json')
    elif refer_dataset == 'refcocog':
        ref_annos_file = os.path.join(args.ref_anno_folder, 'refcocog', 'refs(umd).p')
        ref_instances_file = os.path.join(args.ref_anno_folder, 'refcocog', 'instances.json')
    else:
        raise ValueError(f'Invalid refer dataset and split: {args.refer_split}')

    ref_annos = pickle.load(open(ref_annos_file, 'rb'))
    ref_annos = [x for x in ref_annos if x['split'] == refer_split]
    ref_annos = get_chunk(ref_annos, args.num_chunks, args.chunk_idx)
    ref_instances = json.load(open(ref_instances_file))
    image_id_to_image = {x['id']: x for x in ref_instances['images']}
    image_id_to_instances = {}
    for instance in ref_instances['annotations']:
        image_id = instance['image_id']
        if image_id not in image_id_to_instances:
            image_id_to_instances[image_id] = []
        image_id_to_instances[image_id].append(instance)

    ret = []
    for ref_anno in ref_annos:
        image_id = ref_anno['image_id']
        image = image_id_to_image[image_id]
        ann_id = ref_anno['ann_id']
        instance = [x for x in image_id_to_instances[image_id] if x['id'] == ann_id][0]
        gt_mask = decode_segm(instance['segmentation'], image['height'], image['width'])

        sentences = ref_anno['sentences']
        for sentence in sentences:
            sent_id = sentence['sent_id']
            sent = sentence['sent']
            ret.append({
                'image_id': image_id,
                'image_file': image['file_name'],
                'sent_id': sent_id,
                'sent': sent,
                'gt_mask': gt_mask,
            })

    print(f'Loaded {len(ret)} annotations.')
    return ret


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.get_vision_tower().to(dtype=torch.float16)

    ref_annos = process_ref_anno(args)
    os.makedirs(args.output_folder, exist_ok=True)

    for ref_anno in tqdm(ref_annos):
        image_path = os.path.join(args.image_folder, ref_anno['image_file'])
        attn_path = os.path.join(args.output_folder, f'{ref_anno["sent_id"]}.pth')

        qs = args.template.format(ref_anno['sent'])
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # get answer
        answer = tokenizer.decode(output_ids['sequences'][0], skip_special_tokens=True).strip()

        # automatically detect image tokens
        image_token_start_index = -1
        for i in range(input_ids.shape[1]):
            if input_ids[0, i] == IMAGE_TOKEN_INDEX:
                image_token_start_index = i
                break
        assert image_token_start_index >= 0

        image_token_length = output_ids['attentions'][0][0].shape[-1] - input_ids.shape[1] + 1
        if args.reg_tokens > 0:
            image_token_length -= args.reg_tokens
            image_token_start_index += args.reg_tokens
        image_token_end_index = image_token_start_index + image_token_length
        assert image_token_length == args.feature_height * args.feature_width, \
            f'Image token length mismatch: Expected {args.feature_height * args.feature_width}, got {image_token_length}'

        # process and save attention
        save_sequences = output_ids['sequences'][0].detach().cpu()
        save_attn = []
        for i in range(len(output_ids['attentions'])):
            # n_layers x n_heads x n_output x n_input
            save_attn_i = output_ids['attentions'][i]
            # n_layers x n_heads x n_image_tokens
            save_attn_i = torch.cat([x[:, :, -1, image_token_start_index:image_token_end_index] for x in save_attn_i])
            # feature_height x feature_width
            save_attn_i = save_attn_i.mean(dim=(0, 1)).reshape(args.feature_height, args.feature_width)
            save_attn.append(save_attn_i.detach().float().cpu())

        # n_output x feature_height x feature_width
        save_attn = torch.stack(save_attn)
        save_dict = {
            'answer': answer,
            'sequences': save_sequences,
            'attentions': save_attn,
            'image_id': ref_anno['image_id'],
            'image_file': ref_anno['image_file'],
            'sent': ref_anno['sent'],
            'gt_mask': ref_anno['gt_mask'],
        }
        torch.save(save_dict, attn_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--ref-anno-folder', type=str, default='')
    parser.add_argument('--refer-split', type=str, default='refcoco|val')
    parser.add_argument('--template', type=str, default='Describe the "{}" in the image.')
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
