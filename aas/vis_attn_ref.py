import argparse
import numpy as np
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder', type=str, default='')
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--vis-folder', type=str, default='')
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--maps-per-row', type=int, default=10)
    args = parser.parse_args()

    output_files = sorted([x for x in os.listdir(args.output_folder) if x.endswith('.pth')])
    if args.samples > 0:
        output_files = output_files[:args.samples]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    os.makedirs(args.vis_folder, exist_ok=True)

    for output_file in output_files:
        output_path = os.path.join(args.output_folder, output_file)
        save = torch.load(output_path)
        image_path = os.path.join(args.image_folder, save['image_file'])
        image = Image.open(image_path).convert('RGB')
        image.save(os.path.join(args.vis_folder, output_file.replace('.pth', '_original.jpg')))
        image_width, image_height = image.size

        sequences = save['sequences']
        sequences = sequences[args.offset:]
        attentions = save['attentions'].float()
        attentions = attentions - attentions.mean(dim=0)
        vmin = attentions.min().item()
        vmax = attentions.max().item()

        N = min(sequences.shape[0], attentions.shape[0])
        W = args.maps_per_row
        H = (N + W - 1) // W
        plt.figure(figsize=(W * 2, H * 2))

        for i in range(N):
            token = tokenizer.decode(sequences[i], skip_special_tokens=False)
            attn = attentions[i].numpy()
            plt.subplot(H, W, i + 1)
            plt.imshow(attn, cmap='Reds', interpolation='nearest', vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.title(token, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(args.vis_folder, output_file.replace('.pth', '.png')))
        plt.close()

        N = min(sequences.shape[0], attentions.shape[0])
        W = args.maps_per_row
        H = (N + W - 1) // W
        plt.figure(figsize=(W * 2, H * 2))

        for i in range(N):
            token = tokenizer.decode(sequences[i], skip_special_tokens=False)
            attn = attentions[i].unsqueeze(0).unsqueeze(0)
            image_size = max(image_width, image_height)
            attn = torch.nn.functional.interpolate(attn, (image_size, image_size), mode='bicubic', align_corners=False)
            attn = attn[0, 0, (image_size - image_height) // 2:(image_size + image_height) // 2, (image_size - image_width) // 2:(image_size + image_width) // 2]
            max_indices = torch.argmax(attn.reshape(-1))
            x = max_indices % image_width
            y = max_indices // image_width
            image_vis = image.copy()
            draw = ImageDraw.Draw(image_vis)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill='red')
            plt.subplot(H, W, i + 1)
            plt.imshow(image_vis)
            plt.axis('off')
            plt.title(token, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(args.vis_folder, output_file.replace('.pth', '_point.png')))
        plt.close()
