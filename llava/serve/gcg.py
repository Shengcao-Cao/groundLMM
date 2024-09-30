import argparse
import cv2
import json
import os
import tqdm

import numpy as np
import torch
import spacy

from PIL import Image
from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor

from utils import split_list, get_chunk, group_tokens, encode_segm, decode_segm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str)
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--ref-anno', type=str)
    parser.add_argument('--image-folder', type=str)
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    parser.add_argument('--remove-corner', action='store_true')
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # load models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    spacy_model = spacy.load('en_core_web_lg')
    sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)

    # load reference annotations
    with open(args.ref_anno, 'r') as f:
        ref_anno = json.load(f)
    images = ref_anno['images']
    images = get_chunk(images, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_folder, exist_ok=True)
    for image_index, image in enumerate(tqdm.tqdm(ref_anno['images'])):
        # load image information
        image_id = image['id']
        file_name = image_id + '.jpg'
        file_base = image_id
        image_pil = Image.open(os.path.join(args.image_folder, file_name)).convert('RGB')
        image_width = image_pil.width
        image_height = image_pil.height

        # load saved data
        input_path = os.path.join(args.input_folder, file_base + '.pth')
        save = torch.load(input_path)
        answer = save['answer']
        tokens = save['sequences'][args.offset:]
        attentions = save['attentions'].float()
        attn_mean = attentions.mean(dim=0)
        attentions = attentions - attn_mean

        # group tokens
        groups = group_tokens(tokens, tokenizer, spacy_model)
        for group in groups:
            attns = attentions[group['tokens']]
            attn = attns.mean(dim=0)
            group['attention'] = attn

        if len(groups) == 0:
            continue

        # create segmentation masks
        group_scores = [group['attention'] for group in groups]
        group_scores = torch.stack(group_scores)
        if args.remove_corner:
            min_value = group_scores.min()
            group_scores[:, 0, 0] = min_value
            group_scores[:, 0, -1] = min_value
            group_scores[:, -1, 0] = min_value
            group_scores[:, -1, -1] = min_value

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

        sam_predictor.set_image(np.array(image_pil))
        N, H, W = upsample_scores.shape
        max_indices = torch.argmax(upsample_scores.reshape(N, -1), dim=1)
        h_coords = max_indices // W
        w_coords = max_indices % W
        point_coords_np = torch.stack([w_coords, h_coords], dim=1).numpy()
        point_coords = torch.tensor(sam_predictor.transform.apply_coords(point_coords_np, sam_predictor.original_size)).unsqueeze(1).cuda()
        point_labels = torch.tensor([1] * N).unsqueeze(1).cuda()
        pred_masks, pred_scores, _ = sam_predictor.predict_torch(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        pred_masks = pred_masks[:, -1].cpu()
        pred_scores = pred_scores[:, -1].cpu()

        phrases = [group['phrase'] for group in groups]
        pred_masks = [encode_segm(mask) for mask in pred_masks]

        annotation = {
            'image_id': image_id,
            'caption': answer,
            'phrases': phrases,
            'pred_masks': pred_masks,
        }
        annotation['points'] = [(int(x), int(y)) for x, y in point_coords_np]

        output_path = os.path.join(args.output_folder, file_base + '.json')
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        if args.visualize:
            for group_index in range(len(phrases)):
                phrase = phrases[group_index].replace(' ', '_')
                mask = decode_segm(pred_masks[group_index], image_height, image_width)
                point = annotation['points'][group_index]
                image_mask = np.array(image_pil).copy()
                image_mask[mask == 1] = [255, 0, 0]
                image_mask[point[1]-5:point[1]+6, point[0]-5:point[0]+6] = [255, 255, 255]
                image_mask = Image.fromarray(image_mask)
                image_mask.save(os.path.join(args.output_folder, f'{file_base}_{group_index}_{phrase}.png'))
