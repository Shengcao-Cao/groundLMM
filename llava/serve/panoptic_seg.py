import argparse
import json
import os

import numpy as np
import torch
import spacy

from PIL import Image, ImageDraw, ImageFont
from panopticapi.utils import IdGenerator, id2rgb
from transformers import AutoTokenizer, BertTokenizer, BertModel


def group_tokens(tokens, tokenizer):
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
    doc = spacy_model(sequence)
    for np in doc.noun_chunks:
        phrases.append(np.text)
        phrase_start.append(np.start_char)
        phrase_end.append(np.end_char)

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
            'tokens': group_tokens,
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


def get_bert_embedding(phrase, bert_tokenizer, bert_model, aggregate='mean'):
    phrase = phrase.lower()
    if phrase.startswith('the '):
        phrase = phrase[4:]
    elif phrase.startswith('an '):
        phrase = phrase[3:]
    elif phrase.startswith('a '):
        phrase = phrase[2:]
    with torch.no_grad():
        tokens = bert_tokenizer(phrase, return_tensors='pt', padding=True, truncation=True).to('cuda')
        outputs = bert_model(**tokens)
        embedding = outputs.last_hidden_state.squeeze(dim=0)
        if aggregate == 'mean':
            embedding = embedding.mean(dim=0)
        elif aggregate == 'cls':
            embedding = embedding[0]
        else:
            raise ValueError(f'Invalid aggregation method: {aggregate}')
        return embedding


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
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--merge-by-category', action='store_true')
    parser.add_argument('--category-thresh', type=float, default=-1.0)
    parser.add_argument('--better-visualize', action='store_true')
    args = parser.parse_args()

    # load models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    spacy_model = spacy.load('en_core_web_lg')
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # bert_model = BertModel.from_pretrained('bert-large-uncased').cuda()

    # load reference annotation
    with open(args.ref_anno, 'r') as f:
        ref_anno = json.load(f)
    with open(args.category_file, 'r') as f:
        categories = json.load(f)
    categories = sorted(categories, key=lambda x: x['id'])
    category_dict = {x['id']: x for x in categories}
    category_ids = [x['id'] for x in categories]
    category_names = [x['name'] for x in categories]
    # category_embeddings = [get_bert_embedding(x['name'], bert_tokenizer, bert_model) for x in categories]
    category_embeddings = [get_spacy_embedding(x['name'], spacy_model) for x in categories]
    category_embeddings = torch.stack(category_embeddings)
    print(category_embeddings.shape)

    os.makedirs(args.output_folder, exist_ok=True)
    if args.output_json is None:
        if args.output_folder.endswith('/'):
            args.output_json = args.output_folder[:-1] + '.json'
        else:
            args.output_json = args.output_folder + '.json'

    panoptic_json = []
    id_generator = IdGenerator(category_dict)
    for image_index, image in enumerate(ref_anno['images']):
        # load image information
        image_id = image['id']
        file_name = image['file_name']
        file_base = os.path.splitext(file_name)[0]
        image_width = image['width']
        image_height = image['height']
        pan_segm_id = np.zeros((image_height, image_width), dtype=np.uint32)
        annotation = {
            'image_id': image_id,
            'file_name': file_base + '.png',
        }
        segments_info = []

        # load saved data
        input_path = os.path.join(args.input_folder, file_base + '.pth')
        save = torch.load(input_path)
        answer = save['answer']
        tokens = save['sequences'][1:]
        attentions = save['attentions']
        attn_mean = attentions.mean(dim=0)
        attentions = attentions - attn_mean

        # group tokens
        groups = group_tokens(tokens, tokenizer)
        for group in groups:
            # predict category
            phrase = group['phrase']
            # phrase_embedding = get_bert_embedding(phrase, bert_tokenizer, bert_model)
            phrase_embedding = get_spacy_embedding(phrase, spacy_model)
            similarity = torch.cosine_similarity(category_embeddings, phrase_embedding.unsqueeze(0), dim=1).squeeze()
            most_similar = torch.argmax(similarity)
            most_similar_id = category_ids[most_similar]
            most_similar_name = category_names[most_similar]
            group['pred_category_score'] = similarity[most_similar].item()
            group['pred_category_id'] = most_similar_id
            group['pred_category_name'] = most_similar_name

            attns = attentions[group['tokens']]
            attn = attns.mean(dim=0)
            group['attention'] = attn

        # if args.category_thresh >= 0.0:
        #     groups = [group for group in groups if group['pred_category_score'] >= args.category_thresh]

        # groups = sorted(groups, key=lambda x: x['pred_category_score'], reverse=True)
        # print('#' * 80)
        # for group in groups:
        #     print(f'Phrase: {group["phrase"]}, Most similar: {group["pred_category_name"]}, Similarity: {group["pred_category_score"]}, Category ID: {group["pred_category_id"]}')
        # print('#' * 80)
        # exit(0)

        # create segmentation masks
        # group_scores = [group['pred_category_score'] * group['attention'] for group in groups]
        # group_scores = torch.stack(group_scores)
        group_scores = [group['attention'] for group in groups]
        group_scores = torch.stack(group_scores + [torch.ones_like(attn_mean) * 1e-6])
        upsample_size = max(image_height, image_width)
        crop_h_start = (upsample_size - image_height) // 2
        crop_h_end = crop_h_start + image_height
        crop_w_start = (upsample_size - image_width) // 2
        crop_w_end = crop_w_start + image_width
        upsample_scores = torch.nn.functional.interpolate(group_scores.unsqueeze(0),
                                                          size=(upsample_size, upsample_size),
                                                          mode='bicubic', align_corners=False).squeeze(0)
        upsample_scores = upsample_scores[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        pred_mask_id = torch.argmax(upsample_scores, dim=0)

        if args.merge_by_category:
            detected_category_ids = set([group['pred_category_id'] for group in groups])
            for category_id in detected_category_ids:
                mask = None
                noun_phrases = set()
                for group_index, group in enumerate(groups):
                    if group['pred_category_id'] == category_id and group['pred_category_score'] >= args.category_thresh:
                        if mask is None:
                            mask = (pred_mask_id == group_index)
                        else:
                            mask |= (pred_mask_id == group_index)
                        noun_phrases.add(group['phrase'].lower())
                noun_phrases_str = ', '.join(noun_phrases)
                if mask is None or mask.sum() == 0:
                    continue
                segment_id = id_generator.get_id(category_id)
                panoptic_ann = {
                    'id': segment_id,
                    'category_id': category_id,
                    'category_name': category_dict[category_id]['name'],
                    'noun_phrase': noun_phrases_str,
                }
                segments_info.append(panoptic_ann)
                pan_segm_id[mask] = segment_id
        else:
            for group_index, group in enumerate(groups):
                if group['pred_category_score'] < args.category_thresh:
                    continue
                mask = (pred_mask_id == group_index)
                if mask.sum() == 0:
                    continue
                segment_id = id_generator.get_id(group['pred_category_id'])
                panoptic_ann = {
                    'id': segment_id,
                    'category_id': group['pred_category_id'],
                    'category_name': group['pred_category_name'],
                    'noun_phrase': group['phrase'],
                }
                segments_info.append(panoptic_ann)
                pan_segm_id[mask] = segment_id

        # save panoptic segmentation
        annotation['segments_info'] = segments_info
        panoptic_json.append(annotation)
        Image.fromarray(id2rgb(pan_segm_id)).save(os.path.join(args.output_folder, file_base + '.png'))

        if args.better_visualize:
            original_image = Image.open(os.path.join(args.image_folder, file_name))
            legend_data = []
            for segment in segments_info:
                color = id2rgb(segment['id'])
                legend_data.append({
                    'color': tuple(color),
                    'name': segment['noun_phrase'] + ' -> ' + segment['category_name'],
                })
            legend = draw_legend(legend_data)
            # legend_height = original_image.size[1]
            # legend_width = legend_height * legend.size[0] // legend.size[1]
            legend_width = original_image.size[0] * 2
            legend_height = legend_width * legend.size[1] // legend.size[0]
            legend = legend.resize((legend_width, legend_height))
            save_image = np.concatenate([np.array(original_image), id2rgb(pan_segm_id)], axis=1)
            save_image = np.concatenate([save_image, np.array(legend)], axis=0)
            Image.fromarray(save_image).save(os.path.join(args.output_folder, file_base + '_vis.png'))

        if (image_index + 1) % 100 == 0:
            print(f'Processed {image_index + 1} / {len(ref_anno["images"])} images', flush=True)

    ref_anno['annotations'] = panoptic_json
    with open(args.output_folder + '.json', 'w') as f:
        json.dump(ref_anno, f, indent=2)
