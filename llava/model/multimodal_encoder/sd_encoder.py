from typing import Any, Callable, Dict, List, Optional, Union
import gc
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPVisionModel


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.clone()

        output = {}
        output['up_ft'] = up_ft
        return output


class OneStepSDPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):

        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output


class SDCLIPFeaturizer:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', null_prompt='',
                 clip_model='openai/clip-vit-large-patch14-336',
                 implicit_caption=False, pe=-1):
        vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder")
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        clip_encoder = CLIPVisionModel.from_pretrained(clip_model).to("cuda")

        vae.requires_grad_(False)
        vae.eval()
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        unet.requires_grad_(False)
        unet.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.eval()
        self.clip_encoder = clip_encoder

        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, vae=vae, text_encoder=text_encoder, unet=unet,
                                                         safety_checker=None, low_cpu_mem_usage=False)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        # onestep_pipe.enable_xformers_memory_efficient_attention()
        null_prompt_embeds = onestep_pipe._encode_prompt(
            prompt=null_prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt
        self.pipe = onestep_pipe

        # self.clip_projector and self.clip_pe is created in mm_projector later
        self.implicit_caption = implicit_caption
        self.add_pe = (pe > 0)

    def sd_to_clip_img(self, img_tensor):
        # hard-coded
        # step 1: resize to 14/16
        H, W = img_tensor.shape[-2:]
        H = H * 7 // 8
        W = W * 7 // 8
        img_tensor = nn.functional.interpolate(img_tensor, size=(H, W), mode='bilinear', align_corners=False)
        # step 2: re-normalize
        sd_mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device, dtype=img_tensor.dtype).view(1, 3, 1, 1)
        sd_std = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device, dtype=img_tensor.dtype).view(1, 3, 1, 1)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img_tensor.device, dtype=img_tensor.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img_tensor.device, dtype=img_tensor.dtype).view(1, 3, 1, 1)
        img_tensor = img_tensor * sd_std + sd_mean
        img_tensor = (img_tensor - clip_mean) / clip_std
        return img_tensor

    def forward(self,
                img_tensor,
                prompt='',
                t=261,
                up_ft_indices=[0, 1, 2, 3],
                ensemble_size=8):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        # prepare CLIP image features
        clip_img_tensor = self.sd_to_clip_img(img_tensor)
        clip_features = self.clip_encoder(clip_img_tensor, output_hidden_states=True)
        clip_features = clip_features.hidden_states[-2][:, 1:]
        proj_clip_features = self.clip_projector(clip_features)                 # bs, h * w, c
        proj_clip_features = proj_clip_features.repeat(ensemble_size, 1, 1)     # ensem * bs, h * w, c
        if self.add_pe:
            proj_clip_features = proj_clip_features + self.clip_pe
        if self.implicit_caption:
            prompt_embeds = proj_clip_features
        else:
            if prompt == self.null_prompt:
                prompt_embeds = self.null_prompt_embeds
            else:
                prompt_embeds = self.pipe._encode_prompt(
                    prompt=prompt,
                    device='cuda',
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
            prompt_embeds = prompt_embeds.repeat(ensemble_size * img_tensor.shape[0], 1, 1)
        # produce SD features
        bs = img_tensor.shape[0]
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()           # ensem * bs, c, h, w
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_indices,
            prompt_embeds=prompt_embeds)
        unet_ft = {}
        for up_ft_index in up_ft_indices:
            unet_ft[up_ft_index] = unet_ft_all['up_ft'][up_ft_index]    # ensem * bs, c, h, w
            unet_ft[up_ft_index] = unet_ft[up_ft_index].view(ensemble_size, bs, *unet_ft[up_ft_index].shape[1:])   # ensem, bs, c, h, w
            unet_ft[up_ft_index] = unet_ft[up_ft_index].mean(0, keepdim=False)    # bs, c, h, w
        unet_ft = [unet_ft[i] for i in up_ft_indices]

        bs, c, h, w = unet_ft[1].shape
        clip_features = clip_features.view(bs, h, w, -1)
        clip_features = clip_features.permute(0, 3, 1, 2)       # bs, c, h, w
        unet_ft.append(clip_features)

        return unet_ft


class SDCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.t = args.mm_vision_sd_timestep
        self.ensemble_size = args.mm_vision_sd_ensemble_size

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.args = args
            self.load_model()
        else:
            self.args = args

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor(size=self.args.mm_vision_resolution, crop_size=self.args.mm_vision_resolution,
                                                  mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.vision_tower = SDCLIPFeaturizer(self.vision_tower_name, clip_model=self.args.mm_vision_sd_clip,
                                             implicit_caption=self.args.mm_vision_sd_implicit_caption,
                                             pe=self.args.mm_vision_sd_pe)
        self.is_loaded = True

    def to(self, device=None, dtype=None, *args, **kwargs):
        self.vision_tower.pipe.to(torch_device=device, torch_dtype=dtype)
        self.vision_tower.null_prompt_embeds = self.vision_tower.null_prompt_embeds.to(device=device, dtype=dtype)
        self.vision_tower.clip_encoder.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype, *args, **kwargs)

    def forward(self, images):
        image_features = self.vision_tower.forward(images,
            prompt='', t=self.t, up_ft_indices=[0, 1, 2, 3], ensemble_size=self.ensemble_size)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size[1], device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.pipe.unet.dtype

    @property
    def device(self):
        return self.vision_tower.pipe.unet.device

    @property
    def config(self):
        return self.args

    @property
    def hidden_size(self):
        # hard-coded
        hidden_size = [1280, 1280, 640, 320, 1024]
        return hidden_size

    @property
    def num_patches_per_side(self):
        # hard-coded
        patch_size = 16
        return self.args.mm_vision_resolution // patch_size

    @property
    def num_patches(self):
        # hard-coded
        patch_size = 16
        return (self.args.mm_vision_resolution // patch_size) ** 2
