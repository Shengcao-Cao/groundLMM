import argparse
import numpy as np
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image
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
        image_path = os.path.join(args.image_folder, output_file.replace('.pth', '.jpg'))
        image = Image.open(image_path).convert('RGB')
        image.save(os.path.join(args.vis_folder, output_file.replace('.pth', '_original.jpg')))

        sequences = save['sequences']
        sequences = sequences[args.offset:]
        attentions = save['attentions']
        attentions = attentions - attentions.mean(dim=0)
        # print(sequences.shape, attentions.shape)

        N = min(sequences.shape[0], attentions.shape[0])
        W = args.maps_per_row
        H = (N + W - 1) // W
        plt.figure(figsize=(W * 2, H * 2))

        for i in range(N):
            token = tokenizer.decode(sequences[i], skip_special_tokens=False)
            attn = attentions[i].numpy()
            plt.subplot(H, W, i + 1)
            plt.imshow(attn, cmap='Reds', interpolation='nearest')
            plt.axis('off')
            plt.title(token, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(args.vis_folder, output_file.replace('.pth', '.png')))
        plt.close()
