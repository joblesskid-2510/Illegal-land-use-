"""
Siamese Network for bi-temporal satellite change detection.
Uses a shared encoder (ResNet-18) to extract features from T1 and T2 images,
then a decoder head classifies each pixel as change / no-change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from loguru import logger


class FeatureEncoder(nn.Module):
    """Shared CNN encoder based on ResNet-18, outputs multi-scale features."""

    def __init__(self, in_channels: int = 6, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Modify first conv if input isn't 3 channels
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize by averaging pretrained weights across input channels
            if pretrained:
                with torch.no_grad():
                    pretrained_weight = backbone.conv1.weight
                    # Repeat and average
                    self.conv1.weight[:] = pretrained_weight.mean(dim=1, keepdim=True).repeat(
                        1, in_channels, 1, 1
                    )
        else:
            self.conv1 = backbone.conv1

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 64 channels, /4
        self.layer2 = backbone.layer2  # 128 channels, /8
        self.layer3 = backbone.layer3  # 256 channels, /16
        self.layer4 = backbone.layer4  # 512 channels, /32

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # /2

        x = self.maxpool(x)
        x1 = self.layer1(x)   # /4
        x2 = self.layer2(x1)  # /8
        x3 = self.layer3(x2)  # /16
        x4 = self.layer4(x3)  # /32

        return [x0, x1, x2, x3, x4]


class DifferenceDecoder(nn.Module):
    """
    Decoder that takes feature differences and produces pixel-wise
    change probability map.
    """

    def __init__(self):
        super().__init__()

        # Upsampling path from deepest features
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256 + 256, 128)
        self.up2 = self._up_block(128 + 128, 64)
        self.up1 = self._up_block(64 + 64, 32)
        self.up0 = self._up_block(32 + 64, 16)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def _up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, diffs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            diffs: List of feature difference tensors [d0, d1, d2, d3, d4]
                   from shallowest to deepest

        Returns:
            Change probability map (B, 1, H, W)
        """
        d0, d1, d2, d3, d4 = diffs

        x = self.up4(d4)
        x = self.up3(torch.cat([x, d3], dim=1))
        x = self.up2(torch.cat([x, d2], dim=1))
        x = self.up1(torch.cat([x, d1], dim=1))
        x = self.up0(torch.cat([x, d0], dim=1))

        # Upsample to original resolution (from /2 to /1)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return self.classifier(x)


class SiameseChangeDetector(nn.Module):
    """
    Full Siamese network for change detection.

    Takes two images (before, after) and outputs a per-pixel change map.
    The encoder weights are shared (Siamese), and feature differences
    are decoded into change predictions.
    """

    def __init__(self, in_channels: int = 6, pretrained: bool = True):
        super().__init__()
        self.encoder = FeatureEncoder(in_channels, pretrained)
        self.decoder = DifferenceDecoder()

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"SiameseChangeDetector initialized: {param_count / 1e6:.1f}M params")

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            t1: Before image (B, C, H, W)
            t2: After image (B, C, H, W)

        Returns:
            Change probability map (B, 1, H, W), values in [0, 1] after sigmoid
        """
        # Shared encoder
        feats_t1 = self.encoder(t1)
        feats_t2 = self.encoder(t2)

        # Absolute feature differences at each scale
        diffs = [torch.abs(f2 - f1) for f1, f2 in zip(feats_t1, feats_t2)]

        # Decode
        logits = self.decoder(diffs)

        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Run inference and return binary change mask."""
        self.eval()
        probs = self.forward(t1, t2)
        return (probs > threshold).float()


def build_siamese_model(
    in_channels: int = 6,
    pretrained: bool = True,
    weights_path: str = None,
    device: str = "cpu",
) -> SiameseChangeDetector:
    """Factory function to create and optionally load a Siamese model."""
    model = SiameseChangeDetector(in_channels, pretrained)

    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info(f"Loaded weights from {weights_path}")

    return model.to(device)
