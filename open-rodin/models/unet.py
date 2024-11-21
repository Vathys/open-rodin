"""
# This code is adapted from the guided-diffusion repository by OpenAI.
# The original code can be found at: 
#   - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
# The code has been modified to fit the specific requirements of this project.
"""

from dataclasses import dataclass
from typing import Union, Set, List, Tuple, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf, DictConfig
import hydra

from .base_model import BaseModel
from ..utils.blocks import (
    TimestepEmbedSequential,
    timestep_embedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    AttentionPool2d,
)
from ..utils.misc import conv_nd, zero_module, normalization
from ..utils.fp16_util import convert_module_to_fp16, convert_module_to_fp32


class UNetModel(BaseModel):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    @dataclass
    class UNetConfig(BaseModel.BaseModelConfig):
        image_size: Union[int, Tuple[int, int]] = MISSING
        in_channels: int = MISSING
        out_channels: int = MISSING
        model_channels: int = MISSING
        num_res_blocks: int = MISSING
        attention_resolutions: Union[Set[int], List[int], Tuple[int]] = MISSING
        dropout: float = 0.0
        channel_mult: Union[Set[int], List[int], Tuple[int]] = (1, 2, 4, 8)
        conv_resample: bool = True
        dims: int = 2
        num_classes: Optional[int] = None
        use_checkpoint: bool = False
        use_fp16: bool = False
        num_heads: int = 1
        num_head_channels: Optional[int] = None
        num_heads_upsample: Optional[int] = None
        use_scale_shift_norm: bool = False
        resblock_updown: bool = False
        use_new_attention_order: bool = False

    def _init(self, cfg: UNetConfig):
        if cfg.num_heads_upsample is None:
            self.num_heads_upsample = cfg.num_heads
        else:
            self.num_heads_upsample = cfg.num_heads_upsample

        channel_mult = OmegaConf.to_object(cfg.channel_mult)

        self.dtype = torch.float16 if cfg.use_fp16 else torch.float32

        time_embed_dim = cfg.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(cfg.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if cfg.num_classes is not None:
            self.label_emb = nn.Embedding(cfg.num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * cfg.model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(cfg.dims, cfg.in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        cfg.dropout,
                        out_channels=int(mult * cfg.model_channels),
                        dims=cfg.dims,
                        use_checkpoint=cfg.use_checkpoint,
                        use_scale_shift_norm=cfg.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * cfg.model_channels)
                if ds in cfg.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=cfg.use_checkpoint,
                            num_heads=cfg.num_heads,
                            num_head_channels=cfg.num_head_channels,
                            use_new_attention_order=cfg.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            cfg.dropout,
                            out_channels=out_ch,
                            dims=cfg.dims,
                            use_checkpoint=cfg.use_checkpoint,
                            use_scale_shift_norm=cfg.use_scale_shift_norm,
                            down=True,
                        )
                        if cfg.resblock_updown
                        else Downsample(
                            ch,
                            cfg.conv_resample,
                            dims=cfg.dims,
                            out_channels=out_ch,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                cfg.dropout,
                dims=cfg.dims,
                use_checkpoint=cfg.use_checkpoint,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=cfg.use_checkpoint,
                num_heads=cfg.num_heads,
                num_head_channels=cfg.num_head_channels,
                use_new_attention_order=cfg.use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                cfg.dropout,
                dims=cfg.dims,
                use_checkpoint=cfg.use_checkpoint,
                use_scale_shift_norm=cfg.use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        cfg.dropout,
                        out_channels=int(cfg.model_channels * mult),
                        dims=cfg.dims,
                        use_checkpoint=cfg.use_checkpoint,
                        use_scale_shift_norm=cfg.use_scale_shift_norm,
                    )
                ]
                ch = int(cfg.model_channels * mult)
                if ds in cfg.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=cfg.use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=cfg.num_head_channels,
                            use_new_attention_order=cfg.use_new_attention_order,
                        )
                    )
                if level and i == cfg.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            cfg.dropout,
                            out_channels=out_ch,
                            dims=cfg.dims,
                            use_checkpoint=cfg.use_checkpoint,
                            use_scale_shift_norm=cfg.use_scale_shift_norm,
                            up=True,
                        )
                        if cfg.resblock_updown
                        else Upsample(
                            ch,
                            cfg.conv_resample,
                            dims=cfg.dims,
                            out_channels=out_ch,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(cfg.dims, input_ch, cfg.out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_fp16)
        self.middle_block.apply(convert_module_to_fp16)
        self.output_blocks.apply(convert_module_to_fp16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_fp32)
        self.middle_block.apply(convert_module_to_fp32)
        self.output_blocks.apply(convert_module_to_fp32)

    def _forward(self, data: dict):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = data["image"]
        timesteps = data["timesteps"]
        y = data.get("label", None)

        assert (y is not None) == (
            self.cfg.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.cfg.model_channels))

        if self.cfg.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


@hydra.main(version_base=None, config_path="../configs/model", config_name="unet")
def test_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = UNetModel(cfg=cfg)

    print(model)

    # Test if the model can run
    x = torch.randn(2, 3, 256, 256)
    timesteps = torch.rand(2)

    y = model({"image": x, "timesteps": timesteps})

    # Check the output shape
    print(y.shape)



if __name__ == "__main__":
    test_model()
