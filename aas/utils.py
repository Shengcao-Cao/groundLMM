import math
import numpy as np

import torch

from pycocotools import mask as mask_utils


def split_list(lst, n):
    '''Split a list into n (roughly) equal-sized chunks'''
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def group_tokens(tokens, tokenizer, spacy_model):
    # find correspondence between tokens and chars
    token_start = []
    token_end = []
    cur_length = 0
    for i in range(len(tokens)):
        token_start.append(cur_length)
        sequence = tokenizer.decode(tokens[:i+1], skip_special_tokens=True)
        cur_length = len(sequence)
        token_end.append(cur_length)

    # find noun phrases
    phrases = []
    phrase_start = []
    phrase_end = []
    core_words = []
    doc = spacy_model(sequence)
    for np in doc.noun_chunks:
        phrases.append(np.text)
        core_words.append(np.root.text)
        phrase_start.append(np.root.idx)
        phrase_end.append(np.root.idx + len(np.root.text))

    # group tokens
    groups = []
    for i in range(len(phrases)):
        group_tokens = []
        for j in range(len(tokens)):
            # check if token has overlap with phrase
            if token_start[j] < phrase_end[i] and token_end[j] > phrase_start[i]:
                group_tokens.append(j)
        group = {
            'phrase': phrases[i],
            'core_word': core_words[i],
            'tokens': group_tokens,
            'start_char': phrase_start[i],
            'end_char': phrase_end[i],
        }
        groups.append(group)

    return groups


def get_spacy_embedding(phrase, spacy_model):
    phrase = phrase.lower()
    if phrase.startswith('the '):
        phrase = phrase[4:]
    elif phrase.startswith('an '):
        phrase = phrase[3:]
    elif phrase.startswith('a '):
        phrase = phrase[2:]
    doc = spacy_model(phrase)
    embedding = torch.tensor(doc.vector)
    return embedding


def encode_segm(mask):
    mask = np.array(mask, order='F', dtype=np.uint8)
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def decode_segm(segm, image_height, image_width):
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, image_height, image_width)
        rle = mask_utils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = mask_utils.frPyObjects(segm, image_height, image_width)
    else:
        rle = segm
    mask = mask_utils.decode(rle)
    return mask


def mask_iou(masks):
    N, H, W = masks.shape
    masks = masks.reshape(N, -1).astype(int)
    intersection = masks @ masks.T
    area = masks.sum(axis=1)
    union = area[:, None] + area[None, :] - intersection
    iou = intersection / (union + 1e-6)
    return iou


def nms(masks, iou_thresh=0.75):
    N = masks.shape[0]
    iou = mask_iou(masks)
    keep = [True] * N
    for i in range(N):
        if keep[i]:
            for j in range(i+1, N):
                if iou[i, j] > iou_thresh:
                    keep[j] = False
    return keep


def nms_with_scores(masks, scores, iou_thresh=0.75):
    indices = np.argsort(scores)[::-1]
    masks = masks[indices]
    keep = nms(masks, iou_thresh)
    return indices[keep]


def merge_preds(masks, scores, iou_thresh=0.75):
    N = masks.shape[0]
    iou = mask_iou(masks)
    use_mask = [None] * N
    for i in range(N):
        if use_mask[i] is not None:
            continue
        use_mask[i] = i
        for j in range(N):
            if iou[i, j] > iou_thresh:
                if use_mask[j] is None:
                    use_mask[j] = i
    return use_mask


def convert_mask_SAM(masks, eps=1e-3, edge=256):
    masks = np.clip(masks, eps, 1 - eps)
    masks = np.log(masks / (1 - masks))     # inverse sigmoid
    N, H, W = masks.shape
    masks = torch.tensor(masks).unsqueeze(0)
    if H == W:
        pad_h = 0
        pad_w = 0
        masks = torch.nn.functional.interpolate(masks, (edge, edge), mode='bicubic', align_corners=False)
    elif H > W:
        w = int(W / H * edge)
        pad_w = edge - w
        masks = torch.nn.functional.interpolate(masks, (edge, w), mode='bicubic', align_corners=False)
        masks = torch.nn.functional.pad(masks, (0, pad_w, 0, 0), mode='constant', value=0)
    else:
        h = int(H / W * edge)
        pad_h = edge - h
        masks = torch.nn.functional.interpolate(masks, (h, edge), mode='bicubic', align_corners=False)
        masks = torch.nn.functional.pad(masks, (0, 0, 0, pad_h), mode='constant', value=0)
    return masks.squeeze(0)


def convert_box_SAM(masks, threshold=0.75):
    masks = masks > threshold
    boxes = []
    for mask in masks:
        x, y = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([y.min(), x.min(), y.max() + 1, x.max() + 1])
    return np.array(boxes)
