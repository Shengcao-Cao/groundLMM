import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SDCLIPBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 select_layer=1, clip_proj_in=1024, clip_proj_out=768, concat_clip=False, pe=-1):
        super().__init__()
        self.clip_projector = nn.Sequential(
            nn.Linear(clip_proj_in, clip_proj_out),
            nn.GELU(),
            nn.Linear(clip_proj_out, clip_proj_out),
        )
        self.concat_clip = concat_clip
        self.sd_norm_layer = nn.LayerNorm(in_channels[select_layer])
        self.select_layer = select_layer
        self.sd_proj_layer = nn.Sequential(
            nn.Conv2d(in_channels[select_layer], out_channels // 4, kernel_size=1, stride=1),
            nn.GELU(),
        )
        linear_in = out_channels // 4
        if concat_clip:
            linear_in += clip_proj_in
        self.linear = nn.Sequential(
            nn.Linear(linear_in, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        if pe > 0:
            self.add_pe = True
            self.clip_pe = nn.Parameter(torch.randn(pe, clip_proj_out))
            self.vt_pe = nn.Parameter(torch.randn(pe, out_channels))
        else:
            self.add_pe = False

    def forward(self, x):
        # SD feature
        y = x[self.select_layer]
        y = y.permute(0, 2, 3, 1)
        y = self.sd_norm_layer(y)
        y = y.permute(0, 3, 1, 2)
        y = self.sd_proj_layer(y)

        if self.concat_clip:
            # CLIP feature
            z = x[-1]
            y = torch.cat([y, z], dim=1)

        b, c, h, w = y.shape
        y = y.view(b, c, h * w).permute(0, 2, 1)
        y = self.linear(y)
        if self.add_pe:
            y = y + self.vt_pe
        return y


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'SDCLIPBlock':
        return SDCLIPBlock(config.mm_hidden_size, config.hidden_size,
                           select_layer=config.mm_vision_select_layer,
                           clip_proj_in=config.mm_vision_sd_clip_proj_in,
                           clip_proj_out=config.mm_vision_sd_clip_proj_out,
                           concat_clip=config.mm_vision_sd_concat_clip,
                           pe=config.mm_vision_sd_pe,
                           **kwargs)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
