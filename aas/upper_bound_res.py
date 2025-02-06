import argparse
import json
import os
import pickle
import spacy

import numpy as np
import torch
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pycocotools import mask as mask_utils
from tqdm import tqdm
from transformers import AutoTokenizer


def parallel_apply(func, inputs, workers=16):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(func, inputs), total=len(inputs)))
    return results


def majority_voting(preds, scores, topk=5):
    preds = np.array(preds)
    scores = np.array(scores)
    sort_indices = np.argsort(scores)[::-1][:topk]
    preds = preds[sort_indices]
    scores = scores[sort_indices]
    counts = Counter(preds)
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    if len(candidates) == 1:
        return candidates[0]
    else:
        candidate_scores = {}
        for candidate in candidates:
            candidate_scores[candidate] = scores[preds == candidate].sum()
        return max(candidate_scores, key=candidate_scores.get)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--segmentation', type=str, required=True)
    parser.add_argument('--res-anno', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    spacy_model = spacy.load('en_core_web_lg')

    segm = pickle.load(open(args.segmentation, 'rb'))
    anno = json.load(open(args.res_anno))
    image_id_to_image = {}
    image_id_to_masks = {}
    assert len(anno['images']) == len(segm)
    for image, pred in tqdm(list(zip(anno['images'], segm))):
        image_id_to_image[image['id']] = image
        masks = np.array([mask_utils.decode(x) for x in pred[1]]).astype(bool)
        scores = pred[2]
        sort_indices = np.argsort(scores)[::-1]
        masks = masks[sort_indices]
        image_id_to_masks[image['id']] = masks

    ious = []
    intersections = []
    unions = []

    def process_attn(attn_file):
        save = torch.load(os.path.join(args.input_folder, attn_file))
        gt_mask = save['gt_mask'].astype(bool)
        image_id = save['image_id']
        image = image_id_to_image[image_id]
        image_width = image['width']
        image_height = image['height']

        attentions = save['attentions'].float()
        attn_mean = attentions.mean(dim=0)
        attentions = attentions - attn_mean

        answer = save['answer']
        sequence = save['sequences'][args.offset:]
        sequence = [tokenizer.decode(token) for token in sequence]

        if args.aspect_ratio == 'pad':
            upsample_size = max(image_height, image_width)
            crop_h_start = (upsample_size - image_height) // 2
            crop_h_end = crop_h_start + image_height
            crop_w_start = (upsample_size - image_width) // 2
            crop_w_end = crop_w_start + image_width
            upsample_scores = torch.nn.functional.interpolate(attentions.unsqueeze(0),
                                                              size=(upsample_size, upsample_size),
                                                              mode='bicubic', align_corners=False).squeeze(0)
            upsample_scores = upsample_scores[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        elif args.aspect_ratio == 'original':
            upsample_scores = torch.nn.functional.interpolate(attentions.unsqueeze(0),
                                                              size=(image_height, image_width),
                                                              mode='bicubic', align_corners=False).squeeze(0)
        else:
            raise NotImplementedError(f'Invalid aspect ratio: {args.aspect_ratio}')

        N, H, W = upsample_scores.shape
        max_values, max_indices = torch.max(upsample_scores.reshape(N, -1), dim=1)
        h_coords = max_indices // W
        w_coords = max_indices % W
        point_coords_np = torch.stack([w_coords, h_coords], dim=1).numpy()

        pred_masks = image_id_to_masks[image_id]
        selected_masks = []
        for x, y in point_coords_np:
            selected_mask = None
            for pred_mask_index, pred_mask in enumerate(pred_masks):
                if pred_mask[y, x]:
                    selected_mask = pred_mask_index
                    break
            selected_masks.append(selected_mask)
        max_iou = 0.0
        max_intersection = 0
        max_union = gt_mask.sum()

        # # majority voting
        # voted_mask = majority_voting(selected_masks, max_values, topk=5)
        # if voted_mask is not None:
        #     intersection = (gt_mask & pred_masks[voted_mask]).sum()
        #     union = (gt_mask | pred_masks[voted_mask]).sum()
        #     iou = intersection / union
        #     max_iou = iou
        #     max_intersection = intersection
        #     max_union = union

        # # tokens between the root token and the first ',' or '.'
        # root_token = None
        # for token in spacy_model(answer):
        #     if token.dep_ == 'ROOT':
        #         root_token = token
        #         break
        # root_token_index = -1
        # for i in range(len(sequence)):
        #     if sequence[i] == root_token.text:
        #         root_token_index = i
        #         break
        # punc_token_index = root_token_index + 2
        # for i in range(root_token_index + 2, len(sequence)):
        #     if sequence[i] in [',', '.']:
        #         punc_token_index = i
        #         break
        # selected_mask_index = max_values[root_token_index+1:punc_token_index].argmax() + root_token_index + 1
        # selected_mask = selected_masks[selected_mask_index]

        # token after the first `is`
        selected_mask = None
        for i in range(1, len(sequence)):
            if sequence[i - 1] == 'is':
                selected_mask = selected_masks[i]
                break

        if selected_mask is not None:
            intersection = (gt_mask & pred_masks[selected_mask]).sum()
            union = (gt_mask | pred_masks[selected_mask]).sum()
            iou = intersection / union
            max_iou = iou
            max_intersection = intersection
            max_union = union

        # selected_mask = None
        # for i in range(1, len(sequence)):
        #     if sequence[i - 1] == 'is':
        #         selected_mask = selected_masks[i]
        #         break
        # if selected_mask is not None:
        #     intersection = (gt_mask & pred_masks[selected_mask]).sum()
        #     union = (gt_mask | pred_masks[selected_mask]).sum()
        #     iou = intersection / union
        #     if iou > max_iou:
        #         max_iou = iou
        #         max_intersection = intersection
        #         max_union = union

        # all_good = []
        # for i, selected_mask in enumerate(selected_masks):
        #     if selected_mask is None:
        #         continue
        #     intersection = (gt_mask & pred_masks[selected_mask]).sum()
        #     union = (gt_mask | pred_masks[selected_mask]).sum()
        #     iou = intersection / union
        #     if iou > max_iou:
        #         max_iou = iou
        #         max_intersection = intersection
        #         max_union = union
        #     if iou > 0.8:
        #         all_good.append(i)
        # colored_sequence = [f'\033[92m{sequence[i]}\033[0m' if i in all_good else sequence[i] for i in range(len(sequence))]
        # print(' '.join(colored_sequence))

        return max_iou, max_intersection, max_union

    attn_files = os.listdir(args.input_folder)
    attn_files = [attn_file for attn_file in attn_files if attn_file.endswith('.pth')]
    results = parallel_apply(process_attn, attn_files)
    ious, intersections, unions = zip(*results)

    print(f'gIoU: {np.mean(ious) * 100.0:.2f}')
    print(f'cIoU: {np.mean(intersections) / np.mean(unions) * 100.0:.2f}')
