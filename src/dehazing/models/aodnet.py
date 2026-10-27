"""Checkpoint-compatible PyTorch implementation of the original AOD-Net.

This architecture matches the ``AODnet`` class in
``walsvid/AOD-Net-PyTorch/model.py`` and the Caffe graph in
``Boyiliee/AOD-Net/test/test_template.prototxt``. It is inference-only here:
an instance must be loaded with a verified checkpoint before producing output.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AODNet(nn.Module):
    """The five-convolution AOD-Net architecture from the referenced sources."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)
        self.b = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the AOD-Net output for an RGB tensor in ``[0, 1]``."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("AOD-Net input must have shape (N, 3, H, W)")
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(torch.cat((x1, x2), dim=1)))
        x4 = F.relu(self.conv4(torch.cat((x2, x3), dim=1)))
        k = F.relu(self.conv5(torch.cat((x1, x2, x3, x4), dim=1)))
        output = k * x - k + self.b
        return F.relu(output)


AODnet = AODNet
