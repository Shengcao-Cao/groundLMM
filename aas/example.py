import argparse
import json
import os

import numpy as np
import torch
import spacy

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor

from utils import group_tokens, merge_preds, get_spacy_embedding


# COCO categories, used to filter out abstract noun phrases
seed_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                   'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                   'woman', 'man']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-path', type=str, default='')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--batch-mode', action='store_true')
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--question', type=str, default='Describe the image in detail.')
    parser.add_argument('--conv-mode', type=str, default='llava_v1')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--feature-height', type=int, default=24)
    parser.add_argument('--feature-width', type=int, default=24)
    parser.add_argument('--reg-tokens', type=int, default=0)
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    parser.add_argument('--spacy-score-thresh', type=float, default=0.70)
    parser.add_argument('--sam-score-thresh', type=float, default=0.85)
    args = parser.parse_args()

    # load models
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.get_vision_tower().to(dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    spacy_model = spacy.load('en_core_web_lg')
    sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)

    category_embeddings = [get_spacy_embedding(name, spacy_model) for name in seed_categories]
    category_embeddings = torch.stack(category_embeddings)

    if args.batch_mode:
        image_paths = []
        output_paths = []
        for image_file in sorted(os.listdir(args.image_path)):
            if image_file.endswith('.jpg'):
                image_paths.append(os.path.join(args.image_path, image_file))
                output_paths.append(os.path.join(args.output_path, image_file))
        if args.samples > 0:
            image_paths = image_paths[:args.samples]
            output_paths = output_paths[:args.samples]
    else:
        image_paths = [args.image_path]
        output_paths = [args.output_path]

    os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)

    for image_path, output_path in zip(image_paths, output_paths):
        # process image and question
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

        image = Image.open(image_path).convert('RGB')
        image_width = image.width
        image_height = image.height
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
                    image_token_length = save_attn_i.shape[-1] - input_ids.shape[1] + 1
                    if args.reg_tokens > 0:
                        image_token_length -= args.reg_tokens
                        image_token_start_index += args.reg_tokens
                    image_token_end_index = image_token_start_index + image_token_length
                    assert image_token_length == args.feature_height * args.feature_width, \
                        f'Image token length mismatch: Expected {args.feature_height * args.feature_width}, got {image_token_length}'
                save_attn_i = save_attn_i[:, :, -1, image_token_start_index:image_token_end_index]  # n_layers x n_heads x n_image_tokens
                save_attn_i = save_attn_i.mean(dim=(0, 1))                                          # n_image_tokens
                save_attn_i = save_attn_i.reshape(args.feature_height, args.feature_width)          # feature_height x feature_width
                save_attn.append(save_attn_i.detach().cpu())

            save_attn = torch.stack(save_attn)

            tokens = save_sequences[args.offset:]
            attentions = save_attn.float()
            attn_mean = attentions.mean(dim=0)
            attentions = attentions - attn_mean

            # group tokens
            groups = group_tokens(save_sequences, tokenizer, spacy_model)
            for group in groups:
                attns = attentions[group['tokens']]
                attn = attns.mean(dim=0)
                group['attention'] = attn

            # create segmentation masks
            group_scores = [group['attention'] for group in groups]
            group_scores = torch.stack(group_scores)

            if args.aspect_ratio == 'pad':
                upsample_size = max(image_height, image_width)
                crop_h_start = (upsample_size - image_height) // 2
                crop_h_end = crop_h_start + image_height
                crop_w_start = (upsample_size - image_width) // 2
                crop_w_end = crop_w_start + image_width
                upsample_scores = torch.nn.functional.interpolate(group_scores.unsqueeze(0),
                                                                size=(upsample_size, upsample_size),
                                                                mode='bicubic', align_corners=False).squeeze(0)
                upsample_scores = upsample_scores[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            elif args.aspect_ratio == 'original':
                upsample_scores = torch.nn.functional.interpolate(group_scores.unsqueeze(0),
                                                                size=(image_height, image_width),
                                                                mode='bicubic', align_corners=False).squeeze(0)
            else:
                raise NotImplementedError(f'Invalid aspect ratio: {args.aspect_ratio}')

            sam_predictor.set_image(np.array(image))
            N, H, W = upsample_scores.shape
            max_indices = torch.argmax(upsample_scores.reshape(N, -1), dim=1)
            h_coords = max_indices // W
            w_coords = max_indices % W
            point_coords_np = torch.stack([w_coords, h_coords], dim=1).numpy()
            point_coords = torch.tensor(sam_predictor.transform.apply_coords(point_coords_np, sam_predictor.original_size)).unsqueeze(1).cuda()
            point_labels = torch.tensor([1] * N).unsqueeze(1).cuda()
            pred_masks, pred_scores, _ = sam_predictor.predict_torch(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
            pred_masks = pred_masks[:, -1].cpu().numpy()
            pred_scores = pred_scores[:, -1].cpu().numpy()

            # filter based on spacy similarity and SAM score
            keep_groups = []
            for group_index in range(len(groups)):
                core_word = groups[group_index]['core_word']
                phrase = groups[group_index]['phrase']
                phrase_embedding = get_spacy_embedding(phrase, spacy_model)
                similarity = torch.cosine_similarity(phrase_embedding.unsqueeze(0), category_embeddings, dim=1)
                if similarity.max() < args.spacy_score_thresh:
                    continue
                mask = pred_masks[group_index]
                groups[group_index]['mask'] = mask
                score = pred_scores[group_index]
                if score < args.sam_score_thresh:
                    continue
                keep_groups.append(group_index)
                color = np.random.randint(0, 256, 3)
                groups[group_index]['color'] = color
                area = mask.sum()
                groups[group_index]['area'] = area

            if len(keep_groups) == 0:
                print('#' * 80)
                print(image_path)
                print('-' * 40)
                print(answer)
                print('#' * 80)
                image_ori = Image.open(image_path)
                image_ori.save(output_path.replace('.jpg', '_ori.jpg'))
                continue

            groups = [groups[i] for i in keep_groups]
            pred_masks = pred_masks[keep_groups]
            pred_scores = pred_scores[keep_groups]

            # process overlapping masks for better visualization
            use_mask_indices = merge_preds(pred_masks, pred_scores)

            # produce segmentation map
            image_vis = np.array(image)
            vis_groups = [groups[i] for i in set(use_mask_indices)]
            for group in sorted(vis_groups, key=lambda x: x['area'], reverse=True):
                color = group['color']
                mask = group['mask']
                image_vis[mask > 0] = color

            # use html code to color phrases
            color_answer = answer
            for group_index, group in enumerate(groups):
                core_word = group['core_word']
                start_char = group['start_char']
                end_char = group['end_char']
                color = groups[use_mask_indices[group_index]]['color'].tolist()
                # phrase_html = f'<span style="background-color: rgb({color[0]}, {color[1]}, {color[2]})">{phrase}</span>'
                phrase_html = f'<span style="color: rgb({color[0]}, {color[1]}, {color[2]})">{core_word}</span>'
                new_start = color_answer.find(answer[start_char:])
                color_answer = color_answer[:new_start] + phrase_html + color_answer[new_start + end_char - start_char:]

            # output answers and save visualization
            print('#' * 80)
            print(image_path)
            print('-' * 40)
            print(answer)
            print('-' * 40)
            print(color_answer)
            print('#' * 80)

            image_vis = Image.fromarray(image_vis)
            image_vis.save(output_path)

            image_ori = Image.open(image_path)
            image_ori.save(output_path.replace('.jpg', '_ori.jpg'))
