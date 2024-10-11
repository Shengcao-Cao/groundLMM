import argparse
import json
import os
import tqdm

import numpy as np
import torch
import spacy

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor

from utils import group_tokens, get_spacy_embedding, encode_segm, decode_segm, convert_mask_SAM, convert_box_SAM


def draw_legend(legend_data):
    # Parameters
    box_size = 40          # Size of each color box
    spacing = 10           # Space between boxes and text
    font_size = 20         # Font size for the text
    font_width = 400       # Width of the font
    padding = 20           # Padding around the legend
    items_per_row = 2      # Fixed number of items per row
    # total_width = 400       # Fixed total width for the image
    total_width = (box_size + spacing + font_width) * items_per_row + padding * 2  # Calculate total width dynamically

    # Calculate number of rows needed
    num_items = len(legend_data)
    num_rows = (num_items + items_per_row - 1) // items_per_row  # Round up to the nearest number of rows
    row_height = box_size + spacing  # Height of one row (box + space between rows)
    total_height = num_rows * row_height + padding * 2  # Calculate total height dynamically

    # Create the image with fixed width and variable height
    img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

    # Drawing context
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # Load default font

    # Draw the legend in rows
    x_offset = padding
    y_offset = padding
    for i, item in enumerate(legend_data):
        # Calculate position in the current row
        column = i % items_per_row
        row = i // items_per_row
        
        # Adjust the position for each item
        x_position = x_offset + column * (box_size + spacing + font_width)  # Adjust the spacing between columns
        y_position = y_offset + row * row_height

        # Draw the color box
        draw.rectangle([x_position, y_position, x_position + box_size, y_position + box_size], fill=item["color"])

        # Draw the text
        text_position = (x_position + box_size + spacing, y_position + box_size // 4)
        draw.text(text_position, item["name"], font=font, fill=(0, 0, 0))

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str)
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--output-json', type=str)
    parser.add_argument('--ref-anno', type=str)
    parser.add_argument('--image-folder', type=str)
    parser.add_argument('--category-file', type=str)
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--sam-mode', type=str, default='point')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--more-masks', action='store_true')
    parser.add_argument('--category-thresh', type=float, default=0.0)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    parser.add_argument('--remove-corner', action='store_true')
    parser.add_argument('--attn-temp', type=float, default=0.0002)
    parser.add_argument('--box-thresh', type=float, default=0.75)
    parser.add_argument('--better-visualize', action='store_true')
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()

    # load models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    spacy_model = spacy.load('en_core_web_lg')
    sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)

    # load reference annotation
    with open(args.ref_anno, 'r') as f:
        ref_anno = json.load(f)
    with open(args.category_file, 'r') as f:
        categories = json.load(f)
    categories = [cat for cat in categories if cat['isthing'] == 1]
    categories = sorted(categories, key=lambda x: x['id'])
    category_dict = {x['id']: x for x in categories}
    category_ids = [x['id'] for x in categories]
    category_names = [x['name'] for x in categories]
    category_embeddings = [get_spacy_embedding(x['name'], spacy_model) for x in categories]
    category_embeddings = torch.stack(category_embeddings)

    image_id_to_ref_anno = {}
    for image in ref_anno['images']:
        image_id = image['id']
        image_id_to_ref_anno[image_id] = []
    for obj in ref_anno['annotations']:
        image_id = obj['image_id']
        image_id_to_ref_anno[image_id].append(obj)

    os.makedirs(args.output_folder, exist_ok=True)
    if args.output_json is None:
        if args.output_folder.endswith('/'):
            args.output_json = args.output_folder[:-1] + '.json'
        else:
            args.output_json = args.output_folder + '.json'

    instance_json = []
    id_counter = 0
    for image_index, image in enumerate(tqdm.tqdm(ref_anno['images'])):
        # load image information
        image_id = image['id']
        file_name = image['file_name']
        file_base = os.path.splitext(file_name)[0]
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

        # load gt annotations
        ref_anno_obj = image_id_to_ref_anno[image_id]
        ref_anno_cat_ids = set()
        for obj in ref_anno_obj:
            ref_anno_cat_ids.add(obj['category_id'])
        ref_category_ids = []
        ref_category_names = []
        ref_category_embeddings = []
        for cat_index in range(len(category_ids)):
            if category_ids[cat_index] in ref_anno_cat_ids:
                ref_category_ids.append(category_ids[cat_index])
                ref_category_names.append(category_names[cat_index])
                ref_category_embeddings.append(category_embeddings[cat_index])
        if len(ref_category_ids) == 0:
            continue
        ref_category_embeddings = torch.stack(ref_category_embeddings)

        # group tokens
        groups = group_tokens(tokens, tokenizer, spacy_model)
        for group in groups:
            # predict category
            phrase = group['phrase']
            core_word = group['core_word']
            phrase_embedding = get_spacy_embedding(core_word, spacy_model)
            similarity = torch.cosine_similarity(ref_category_embeddings, phrase_embedding.unsqueeze(0), dim=1)
            most_similar = torch.argmax(similarity)
            most_similar_id = ref_category_ids[most_similar]
            most_similar_name = ref_category_names[most_similar]
            group['pred_category_score'] = similarity[most_similar].item()
            group['pred_category_id'] = most_similar_id
            group['pred_category_name'] = most_similar_name

            attns = attentions[group['tokens']]
            attn = attns.mean(dim=0)
            group['attention'] = attn

        groups = [group for group in groups if group['pred_category_score'] >= args.category_thresh]
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
        if args.sam_mode == 'point':
            max_indices = torch.argmax(upsample_scores.reshape(N, -1), dim=1)
            h_coords = max_indices // W
            w_coords = max_indices % W
            point_coords_np = torch.stack([w_coords, h_coords], dim=1).numpy()
            point_coords = torch.tensor(sam_predictor.transform.apply_coords(point_coords_np, sam_predictor.original_size)).unsqueeze(1).cuda()
            point_labels = torch.tensor([1] * N).unsqueeze(1).cuda()
            pred_masks, pred_scores, _ = sam_predictor.predict_torch(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        elif args.sam_mode == 'mask':
            mask_input = convert_mask_SAM(upsample_scores / args.attn_temp).unsqueeze(1).cuda()
            boxes_np = convert_box_SAM(upsample_scores / args.attn_temp, threshold=args.box_thresh)
            boxes = torch.tensor(sam_predictor.transform.apply_boxes(boxes_np, sam_predictor.original_size)).unsqueeze(1).cuda()
            pred_masks, pred_scores, _ = sam_predictor.predict_torch(point_coords=None, point_labels=None, mask_input=mask_input, boxes=boxes, multimask_output=True)
        else:
            raise NotImplementedError(f'Invalid SAM mode: {args.sam_mode}')

        if args.more_masks:
            pred_masks = pred_masks.flatten(0, 1).cpu()
            pred_scores = pred_scores.flatten(0, 1).cpu()
        else:
            pred_masks = pred_masks[:, -1].cpu()
            pred_scores = pred_scores[:, -1].cpu()

        instances = []
        for group_index, group in enumerate(groups):
            if group['pred_category_score'] < args.category_thresh:
                continue
            if args.more_masks:
                for i in range(3):
                    mask = pred_masks[group_index * 3 + i]
                    if mask.sum() == 0:
                        continue
                    id_counter += 1
                    segm_rle = encode_segm(mask.numpy())
                    instance = {
                        'id': id_counter,
                        'image_id': image_id,
                        'category_id': group['pred_category_id'],
                        'category_name': group['pred_category_name'],
                        'noun_phrase': group['phrase'],
                        'segmentation': segm_rle,
                        'score': pred_scores[group_index * 3 + i].item(),
                        'iscrowd': 0,
                        'area': mask.sum().item(),
                    }
                    if args.sam_mode == 'point':
                        instance['point'] = [int(point_coords_np[group_index][0]), int(point_coords_np[group_index][1])]
                    elif args.sam_mode == 'mask':
                        instance['box'] = [int(x) for x in boxes_np[group_index * 3 + i].tolist()]
                    instances.append(instance)
            else:
                mask = pred_masks[group_index]
                if mask.sum() == 0:
                    continue
                id_counter += 1
                segm_rle = encode_segm(mask.numpy())
                instance = {
                    'id': id_counter,
                    'image_id': image_id,
                    'category_id': group['pred_category_id'],
                    'category_name': group['pred_category_name'],
                    'noun_phrase': group['phrase'],
                    'segmentation': segm_rle,
                    'score': pred_scores[group_index].item(),
                    'iscrowd': 0,
                    'area': mask.sum().item(),
                }
                if args.sam_mode == 'point':
                    instance['point'] = [int(point_coords_np[group_index][0]), int(point_coords_np[group_index][1])]
                elif args.sam_mode == 'mask':
                    instance['box'] = [int(x) for x in boxes_np[group_index].tolist()]
                instances.append(instance)

        instance_json.extend(instances)

        if args.better_visualize:
            original_image = Image.open(os.path.join(args.image_folder, file_name))
            segm_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            legend_data = []
            for instance in instances:
                mask = decode_segm(instance['segmentation'], image_height, image_width)
                color = category_dict[instance['category_id']]['color']
                segm_map[mask > 0] = color
                legend_data.append({
                    'color': tuple(color),
                    'name': instance['noun_phrase'] + ' -> ' + instance['category_name'],
                })
            gt_segm_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            for obj in ref_anno_obj:
                if obj['image_id'] == image_id:
                    mask = decode_segm(obj['segmentation'], image_height, image_width)
                    color = category_dict[obj['category_id']]['color']
                    gt_segm_map[mask > 0] = color
            legend = draw_legend(legend_data)
            legend_width = original_image.size[0] * 3
            legend_height = legend_width * legend.size[1] // legend.size[0]
            legend = legend.resize((legend_width, legend_height))
            save_image = np.concatenate([np.array(original_image), gt_segm_map, segm_map], axis=1)
            save_image = np.concatenate([save_image, np.array(legend)], axis=0)
            Image.fromarray(save_image).save(os.path.join(args.output_folder, file_base + '_vis.png'))

            for instance in instances:
                individual_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                color = category_dict[instance['category_id']]['color']
                mask = decode_segm(instance['segmentation'], image_height, image_width)
                individual_map[mask > 0] = color
                if 'point' in instance:
                    point = instance['point']
                    individual_map[point[1]-5:point[1]+6, point[0]-5:point[0]+6] = (255, 255, 255)
                if 'box' in instance:
                    x1, y1, x2, y2 = instance['box']
                    for x in range(x1, x2):
                        individual_map[y1, x] = (255, 255, 255)
                        individual_map[y2 - 1, x] = (255, 255, 255)
                    for y in range(y1, y2):
                        individual_map[y, x1] = (255, 255, 255)
                        individual_map[y, x2 - 1] = (255, 255, 255)
                individual_save_image = np.concatenate([np.array(original_image), individual_map], axis=1)
                output_file = os.path.join(args.output_folder, file_base + f'_{instance["id"]}_{instance["noun_phrase"].replace(" ", "_")}.png')
                Image.fromarray(individual_save_image).save(output_file)

            for obj in ref_anno_obj:
                individual_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                mask = decode_segm(obj['segmentation'], image_height, image_width)
                color = category_dict[obj['category_id']]['color']
                individual_map[mask > 0] = color
                individual_save_image = np.concatenate([np.array(original_image), individual_map], axis=1)
                output_file = os.path.join(args.output_folder, file_base + f'_gt_{category_dict[obj["category_id"]]["name"]}.png')
                Image.fromarray(individual_save_image).save(output_file)

        if args.samples is not None and image_index + 1 >= args.samples:
            break

    with open(args.output_json, 'w') as f:
        json.dump(instance_json, f, indent=2)
