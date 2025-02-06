import argparse
import json
import os
import pickle

import numpy as np
import torch
import spacy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pycocotools import mask as mask_utils
from tqdm import tqdm
from transformers import AutoTokenizer


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def parallel_apply(func, inputs, workers=16):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(func, inputs), total=len(inputs)))
    return results


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

        # phrase = save['sent']
        # # phrase_embedding = spacy_model(phrase).vector
        # for phrase_part in spacy_model(phrase):
        #     if phrase_part.dep_ == 'ROOT':
        #         phrase_embedding = phrase_part.vector
        #         break

        # sequence = save['sequences'][args.offset:]
        # sequence = [tokenizer.decode(token) for token in sequence]
        # token_embeddings = [spacy_model(token).vector for token in sequence]

        # similarities = np.array([cosine_similarity(phrase_embedding, token_embedding) for token_embedding in token_embeddings])
        # similarities = np.maximum(similarities, 0.0)

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

        # upsample_scores = upsample_scores * similarities.reshape(-1, 1, 1)
        upsample_scores = upsample_scores[:8].amax(dim=0)

        H, W = upsample_scores.shape
        max_indices = torch.argmax(upsample_scores.reshape(-1))
        y = max_indices // W
        x = max_indices % W

        pred_masks = image_id_to_masks[image_id]
        selected_mask = None
        for pred_mask in pred_masks:
            if pred_mask[y, x]:
                selected_mask = pred_mask
                break
        if selected_mask is not None:
            intersection = (gt_mask & selected_mask).sum()
            union = (gt_mask | selected_mask).sum()
            iou = intersection / union
        else:
            iou = 0.0
            intersection = 0
            union = gt_mask.sum()

        return iou, intersection, union

    attn_files = os.listdir(args.input_folder)
    attn_files = [attn_file for attn_file in attn_files if attn_file.endswith('.pth')]
    results = parallel_apply(process_attn, attn_files)
    ious, intersections, unions = zip(*results)

    print(f'gIoU: {np.mean(ious) * 100.0:.2f}')
    print(f'cIoU: {np.mean(intersections) / np.mean(unions) * 100.0:.2f}')
