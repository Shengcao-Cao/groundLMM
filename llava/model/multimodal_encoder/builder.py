import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .convnext_clip_encoder import ConvNeXtCLIPVisionTower
from .siglip_encoder import SigLIPVisionTower
from .dinov2_encoder import DINOv2VisionTower
from .timm_siglip_encoder import TimmSigLIPVisionTower
from .sd_encoder import SDCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    # OpenAI CLIP
    if vision_tower.startswith('openai') and 'clip' in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # OpenCLIP
    elif vision_tower.startswith('laion') and 'CLIP' in vision_tower:
        return ConvNeXtCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # SigLIP
    elif vision_tower.startswith('google') and 'siglip' in vision_tower:
        return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # SigLIP (timm implementation)
    elif vision_tower.startswith('timm') and 'SigLIP' in vision_tower:
        return TimmSigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # DINOv2 (timm implementation)
    elif vision_tower.startswith('timm') and 'dinov2' in vision_tower:
        return DINOv2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("stabilityai") or vision_tower.startswith("runwayml") or vision_tower.startswith("stable-diffusion"):
        return SDCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
