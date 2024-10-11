import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
import open_clip


DEFAULT_IMAGE_PROCESSOR = 'openai/clip-vit-large-patch14-336'


def interpolate_pos_encoding(position_embedding: torch.Tensor, height: int, width: int) -> torch.Tensor:
    import math

    num_patches = height * width
    num_positions = position_embedding.shape[0]
    if num_patches == num_positions and height == width:
        return position_embedding

    dim = position_embedding.shape[-1]
    patch_pos_embed = position_embedding.reshape(1, int(math.sqrt(num_positions + 0.1)), int(math.sqrt(num_positions + 0.1)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=((height + 0.1) / math.sqrt(num_positions), (width + 0.1) / math.sqrt(num_positions)),
        mode='bicubic',
        align_corners=False,
    )
    if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
        raise ValueError('Width or height does not match with the interpolated position embeddings')

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(-1, dim)
    return patch_pos_embed


class TimmSigLIPVisionTower(nn.Module):
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

        num_patches_per_side = self.resolution // 14
        embed = self.vision_tower.pos_embed.data[0]
        num_pos_embed = embed.shape[0]
        if num_patches_per_side * num_patches_per_side != num_pos_embed:
            embed = interpolate_pos_encoding(
                embed,
                num_patches_per_side,
                num_patches_per_side,
            )
            self.vision_tower.pos_embed.data = embed.unsqueeze(0)
            self.vision_tower.patch_embed.img_size = (self.resolution, self.resolution)
            self.vision_tower.patch_embed.grid_size = (num_patches_per_side, num_patches_per_side)
            self.vision_tower.patch_embed.num_patches = num_patches_per_side * num_patches_per_side

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        images = images.to(device=self.device, dtype=self.dtype)
        bs = images.shape[0]
        image_features = self.vision_tower.forward_features(images)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.vision_tower.patch_embed.proj.weight.device
