import argparse
import numpy as np
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')
tokens = tokenizer.encode('The image depicts a group of people gathered around a bar, with a man standing behind the counter serving wine. There are several people in the scene, some standing closer to the bar and others further away. A man in a hat is standing near the counter, possibly waiting to be served or engaging in conversation with the bartender.')
tokens = [tokenizer.decode(token) for token in tokens]

for attn_file in os.listdir('.'):
    if not attn_file.endswith('.pth'):
        continue
    attentions = torch.load(attn_file).float().cpu()
    attentions = attentions.mean(dim=0).transpose(0, 1).reshape(77, 24, 24)
    attentions = attentions - attentions.mean(dim=0)
    vmin = attentions.min().item()
    vmax = attentions.max().item()

    N = attentions.shape[0]
    W = 5
    H = (N + W - 1) // W
    plt.figure(figsize=(W * 2, H * 2))

    for i in range(N):
        if i >= len(tokens):
            break
        token = tokens[i]
        attn = attentions[i].numpy()
        plt.subplot(H, W, i + 1)
        plt.imshow(attn, cmap='Reds', interpolation='nearest')
        plt.axis('off')
        plt.title(token, fontsize=8)

    plt.tight_layout()
    plt.savefig(attn_file.replace('.pth', '.pdf'))
    plt.savefig(attn_file.replace('.pth', '.png'))
    plt.close()
