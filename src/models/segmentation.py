"""
Semantic segmentation model for land cover classification.
Uses DeepLabV3+ with a ResNet-50 backbone via segmentation_models_pytorch.
6-class output: building, road, vegetation, water, bare_soil, construction.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from loguru import logger


# Class definitions
LAND_COVER_CLASSES = {
    0: {"name": "background", "color": [0, 0, 0]},
    1: {"name": "building", "color": [255, 0, 0]},
    2: {"name": "road", "color": [128, 128, 128]},
    3: {"name": "vegetation", "color": [0, 255, 0]},
    4: {"name": "water", "color": [0, 0, 255]},
    5: {"name": "bare_soil", "color": [210, 180, 140]},
    6: {"name": "construction", "color": [255, 165, 0]},
}

CLASS_NAMES = [LAND_COVER_CLASSES[i]["name"] for i in range(7)]
CLASS_COLORS = [LAND_COVER_CLASSES[i]["color"] for i in range(7)]


class LandCoverSegmenter(nn.Module):
    """
    DeepLabV3+ semantic segmentation model for land cover classification.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        in_channels: int = 6,
        num_classes: int = 7,
        pretrained: bool = True,
    ):
        super().__init__()

        encoder_weights = "imagenet" if pretrained else None

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Raw logits, we apply softmax in inference
        )

        self.num_classes = num_classes

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LandCoverSegmenter initialized: {encoder_name}, "
            f"{num_classes} classes, {param_count / 1e6:.1f}M params"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            Class logits (B, num_classes, H, W)
        """
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference, return class index per pixel."""
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=1)  # (B, H, W)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference, return softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def colorize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a class index mask to an RGB visualization.

    Args:
        mask: (H, W) tensor with class indices

    Returns:
        (3, H, W) uint8 tensor
    """
    import numpy as np

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_np = mask.cpu().numpy()

    for cls_id, info in LAND_COVER_CLASSES.items():
        rgb[mask_np == cls_id] = info["color"]

    return torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)


def build_segmentation_model(
    encoder_name: str = "resnet50",
    in_channels: int = 6,
    num_classes: int = 7,
    pretrained: bool = True,
    weights_path: str = None,
    device: str = "cpu",
) -> LandCoverSegmenter:
    """Factory function to create and optionally load a segmentation model."""
    model = LandCoverSegmenter(encoder_name, in_channels, num_classes, pretrained)

    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        logger.info(f"Loaded segmentation weights from {weights_path}")

    return model.to(device)
