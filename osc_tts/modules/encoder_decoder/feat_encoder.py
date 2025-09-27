import torch
import torch.nn as nn

from typing import List

from ..blocks.vocos import VocosBackbone
from ..blocks.samper import SamplingBlock


class Encoder(nn.Module):
    """Encoder module with convnext and downsampling blocks"""

    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,
        sample_ratios: List[int] = [1, 1],
    ):
        super().__init__()
        """
        Encoder module with VocosBackbone and sampling blocks.

        Args:
            sample_ratios (List[int]): sample ratios
                example: [2, 2] means downsample by 2x and then upsample by 2x
        """
        self.encoder = VocosBackbone(
            input_channels=input_channels,
            dim=vocos_dim,
            intermediate_dim=vocos_intermediate_dim,
            num_layers=vocos_num_layers,
            condition_dim=None,
        )

        modules = [
            nn.Sequential(
                SamplingBlock(
                    dim=vocos_dim,
                    groups=vocos_dim,
                    downsample_scale=ratio,
                ),
                VocosBackbone(
                    input_channels=vocos_dim,
                    dim=vocos_dim,
                    intermediate_dim=vocos_intermediate_dim,
                    num_layers=2,
                    condition_dim=None,
                ),
            )
            for ratio in sample_ratios
        ]

        self.downsample = nn.Sequential(*modules)

        self.project = nn.Linear(vocos_dim, out_channels)

    def forward(self, x: torch.Tensor, *args):
        """
        Args:
            x (torch.Tensor): (batch_size, input_channels, length)

        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.encoder(x)
        x = self.downsample(x)
        x = self.project(x)
        return x.transpose(1, 2)
