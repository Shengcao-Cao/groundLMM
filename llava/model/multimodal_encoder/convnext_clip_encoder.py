import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
import open_clip


DEFAULT_IMAGE_PROCESSOR = 'openai/clip-vit-large-patch14-336'


class ConvNeXtCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.resolution = getattr(args, 'mm_vision_resolution', 336)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(DEFAULT_IMAGE_PROCESSOR)
        self.image_processor.size['shortest_edge'] = self.resolution
        self.image_processor.crop_size['width'] = self.resolution
        self.image_processor.crop_size['height'] = self.resolution

        model = open_clip.create_model('hf-hub:' + self.vision_tower_name, device='cuda')
        self.vision_tower = model.visual.trunk
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        images = images.to(device=self.device, dtype=self.dtype)
        bs = images.shape[0]
        image_features = self.vision_tower.forward_features(images)
        features_height = image_features.shape[2]
        features_width = image_features.shape[3]
        image_features = image_features.view(bs, -1, features_height * features_width).permute(0, 2, 1)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.stem[0].weight.dtype

    @property
    def device(self):
        return self.vision_tower.stem[0].weight.device
