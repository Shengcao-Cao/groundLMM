import argparse
import json
import os
import tqdm

import numpy as np
import torch

from PIL import Image

from utils import encode_segm, decode_segm


# copied from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


# copied from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


# copied from https://github.com/wusize/F-LMM/blob/main/scripts/multiprocess_eval_png.py
def average_accuracy(ious):
    accuracy = []
    average_acc = 0
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (ious >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)

        accuracy.append(a)
    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])):
        average_acc += (np.abs(t[1] - t[0])) * accuracy[i]

    return average_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str)
    parser.add_argument('--output-json', type=str)
    parser.add_argument('--panoptic-anno', type=str)
    parser.add_argument('--png-anno', type=str)
    parser.add_argument('--panoptic-pred-folder', type=str)
    parser.add_argument('--image-folder', type=str)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    parser.add_argument('--group-aggregation', type=str, default='max')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # load png annotations
    with open(args.panoptic_anno, 'r') as f:
        pan_anno = json.load(f)
    with open(args.png_anno, 'r') as f:
        png_anno = json.load(f)

    image_id_to_image = {}
    for image in pan_anno['images']:
        image_id_to_image[image['id']] = image
    image_id_to_pan_anno = {}
    for anno in pan_anno['annotations']:
        image_id_to_pan_anno[anno['image_id']] = anno
    category_id_to_category = {}
    for category in pan_anno['categories']:
        category_id_to_category[category['id']] = category
    annotation_id_to_annotation = {}
    for image_anno in pan_anno['annotations']:
        for anno in image_anno['segments_info']:
            annotation_id_to_annotation[anno['id']] = anno

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    if args.visualize:
        output_folder = args.output_json.replace('.json', '')
        os.makedirs(output_folder, exist_ok=True)

    iou_list = []
    isplural_list = []
    isthing_list = []
    for png_index in tqdm.tqdm(range(len(png_anno))):
        # load image information
        image_id = int(png_anno[png_index]['image_id'])
        image = image_id_to_image[image_id]
        file_name = image['file_name']
        file_base = f'{png_index:012d}'
        image_pil = Image.open(os.path.join(args.image_folder, file_name)).convert('RGB')
        image_width = image_pil.width
        image_height = image_pil.height

        # load saved data
        input_path = os.path.join(args.input_folder, file_base + '.pth')
        save = torch.load(input_path)
        answer = save['answer']
        tokens = save['sequences']
        token_groups = save['token_groups']
        attentions = save['attentions'].float()
        attn_mean = attentions.mean(dim=0)
        attentions = attentions - attn_mean

        # load png data
        pan_image_path = os.path.join(args.panoptic_anno.replace('.json', ''), file_name.replace('.jpg', '.png'))
        pan_image = np.array(Image.open(pan_image_path))
        pan_id_map = rgb2id(pan_image)

        pred_pan_image_path = os.path.join(args.panoptic_pred_folder, file_name.replace('.jpg', '.png'))
        pred_pan_image = np.array(Image.open(pred_pan_image_path))
        pred_pan_id_map = rgb2id(pred_pan_image)

        segments = png_anno[png_index]['segments']
        assert len(segments) == len(token_groups)
        groups = []

        for segment_index, segment in enumerate(segments):
            utterance = segment['utterance']
            group_tokens = tokens[token_groups[segment_index]]
            if len(segment['segment_ids']) > 0:
                isplural = segment['plural']
                isnoun = segment['noun']
                assert isnoun
                if len(segment['segment_ids']) == 1:
                    segment_id = segment['segment_ids'][0]
                    segment_anno = annotation_id_to_annotation[int(segment_id)]
                    segment_category = category_id_to_category[segment_anno['category_id']]
                    isthing = segment_category['isthing']
                    gt_mask = (pan_id_map == segment_anno['id'])
                else:
                    assert isplural
                    isthing = True
                    gt_mask = np.zeros_like(pan_id_map, dtype=bool)
                    for segment_id in segment['segment_ids']:
                        segment_anno = annotation_id_to_annotation[int(segment_id)]
                        gt_mask |= (pan_id_map == segment_anno['id'])
                segment['isthing'] = isthing

                assert gt_mask.sum() > 0

                if args.group_aggregation == 'max':
                    group_attention = attentions[token_groups[segment_index]].amax(dim=0)
                elif args.group_aggregation == 'mean':
                    group_attention = attentions[token_groups[segment_index]].mean(dim=0)
                elif args.group_aggregation == 'first':
                    group_attention = attentions[token_groups[segment_index]][0]
                group = {
                    'phrase': utterance,
                    'tokens': group_tokens,
                    'isplural': isplural,
                    'isthing': isthing,
                    'gt_mask': gt_mask,
                    'attention': group_attention,
                }
                groups.append(group)
                segment['gt_mask'] = gt_mask
                segment['pd_mask'] = None

        if len(groups) == 0:
            continue

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

        N, H, W = upsample_scores.shape
        max_indices = torch.argmax(upsample_scores.reshape(N, -1), dim=1)
        h_coords = max_indices // W
        w_coords = max_indices % W
        point_coords_np = torch.stack([w_coords, h_coords], dim=1).numpy()

        pred_masks = []
        for x, y in point_coords_np:
            mask_id = pred_pan_id_map[y, x]
            mask = (pred_pan_id_map == mask_id)
            pred_masks.append(mask)

        phrases = [group['phrase'] for group in groups]
        pred_masks = [encode_segm(mask) for mask in pred_masks]

        annotation = {
            'image_id': image_id,
            'caption': answer,
            'phrases': phrases,
            'pred_masks': pred_masks,
        }
        annotation['points'] = [(int(x), int(y)) for x, y in point_coords_np]

        count = 0
        for segment in segments:
            if 'pd_mask' in segment:
                pd_mask = decode_segm(pred_masks[count], image_height, image_width).astype(bool)
                gt_mask = segment['gt_mask']
                segment['iou'] = float(((pd_mask & gt_mask).sum() / (pd_mask | gt_mask).sum()).item())
                del segment['pd_mask']
                del segment['gt_mask']
                segment['pd_mask'] = pred_masks[count]
                count += 1
                iou_list.append(segment['iou'])
                isplural_list.append(segment['plural'])
                isthing_list.append(segment['isthing'])
        assert count == len(groups), f'{count} != {len(groups)}'

        if args.visualize:
            for group_index in range(len(phrases)):
                phrase = phrases[group_index].replace(' ', '_')
                mask = decode_segm(pred_masks[group_index], image_height, image_width)
                point = annotation['points'][group_index]
                image_mask = np.array(image_pil).copy()
                image_mask[mask == 1] = [255, 0, 0]
                image_mask[point[1]-5:point[1]+6, point[0]-5:point[0]+6] = [255, 255, 255]
                image_mask = Image.fromarray(image_mask)
                image_mask.save(os.path.join(output_folder, f'{file_base}_{group_index}_{phrase}.png'))

    iou_list = np.array(iou_list)
    isplural_list = np.array(isplural_list).astype(bool)
    isthing_list = np.array(isthing_list).astype(bool)
    AA = average_accuracy(iou_list)
    # AA_singular = average_accuracy(iou_list[~isplural_list])
    # AA_plural = average_accuracy(iou_list[isplural_list])
    AA_thing = average_accuracy(iou_list[isthing_list])
    AA_stuff = average_accuracy(iou_list[~isthing_list])
    print(f'Average Accuracy: {AA:.4f}')
    # print(f'Average Accuracy Singular: {AA_singular:.4f}')
    # print(f'Average Accuracy Plural: {AA_plural:.4f}')
    print(f'Average Accuracy Thing: {AA_thing:.4f}')
    print(f'Average Accuracy Stuff: {AA_stuff:.4f}')

    with open(args.output_json, 'w') as f:
        json.dump(png_anno, f)
