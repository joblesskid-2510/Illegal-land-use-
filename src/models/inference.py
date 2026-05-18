"""
Unified inference pipeline.
Runs change detection and segmentation on preprocessed tile pairs,
stitches results back to georeferenced GeoTIFFs.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from src.ingestion.preprocessor import preprocessor
from src.models.siamese_net import SiameseChangeDetector, build_siamese_model
from src.models.segmentation import LandCoverSegmenter, build_segmentation_model


class InferencePipeline:
    """Orchestrates model inference over tiled satellite imagery."""

    def __init__(
        self,
        change_model: Optional[SiameseChangeDetector] = None,
        seg_model: Optional[LandCoverSegmenter] = None,
        device: str = None,
        batch_size: int = None,
        confidence_threshold: float = None,
    ):
        self.device = device or settings.model.device
        self.batch_size = batch_size or settings.model.batch_size
        self.confidence_threshold = confidence_threshold or settings.model.confidence_threshold

        self.change_model = change_model
        self.seg_model = seg_model

    def load_models(
        self,
        change_weights: str = None,
        seg_weights: str = None,
    ):
        """Load both models from weights or pretrained."""
        if self.change_model is None:
            self.change_model = build_siamese_model(
                in_channels=6,  # Sentinel-2 change detection bands
                pretrained=True,
                weights_path=change_weights or settings.model.siamese_weights_path,
                device=self.device,
            )
        if self.seg_model is None:
            self.seg_model = build_segmentation_model(
                in_channels=6,
                num_classes=7,
                pretrained=True,
                weights_path=seg_weights or settings.model.seg_weights_path,
                device=self.device,
            )
        logger.info("Models loaded")

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy tile to model-ready tensor."""
        return torch.from_numpy(array).float().to(self.device)

    def _run_batched(
        self,
        model_fn,
        *tile_lists: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run a model function over batched tiles."""
        results = []
        n = len(tile_lists[0])

        for i in range(0, n, self.batch_size):
            batch_end = min(i + self.batch_size, n)
            batches = []
            for tl in tile_lists:
                batch = torch.stack([
                    self._to_tensor(tl[j]) for j in range(i, batch_end)
                ])
                batches.append(batch)

            with torch.no_grad():
                output = model_fn(*batches)

            # Convert to numpy
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            for j in range(output.shape[0]):
                results.append(output[j])

        return results

    def detect_changes(
        self,
        t1_path: Path,
        t2_path: Path,
        output_path: Optional[Path] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Run change detection on a bi-temporal image pair.

        Args:
            t1_path: Before-image GeoTIFF
            t2_path: After-image GeoTIFF
            output_path: Optional path to save change map GeoTIFF

        Returns:
            Tuple of (change_map array, metadata)
        """
        self.load_models()

        # Preprocess
        t1_tiles, t2_tiles, meta = preprocessor.prepare_pair(t1_path, t2_path)

        t1_data = [t["data"] for t in t1_tiles]
        t2_data = [t["data"] for t in t2_tiles]

        logger.info(f"Running change detection on {len(t1_data)} tiles...")

        # Run model
        predictions = self._run_batched(
            self.change_model.forward,
            t1_data,
            t2_data,
        )

        # Squeeze channel dim: (1, H, W) -> (H, W)
        predictions = [p.squeeze(0) for p in predictions]

        # Stitch
        change_map = preprocessor.stitch_tiles(t1_tiles, predictions, meta, num_classes=1)

        # Save
        if output_path:
            preprocessor.save_geotiff(change_map, output_path, meta)

        logger.info(
            f"Change detection complete. "
            f"Changed pixels: {(change_map > self.confidence_threshold).sum()} / {change_map.size}"
        )

        return change_map, meta

    def segment_landcover(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Run semantic segmentation on a single image.

        Args:
            image_path: GeoTIFF to segment
            output_path: Optional path to save segmentation GeoTIFF

        Returns:
            Tuple of (class_map array [H, W], metadata)
        """
        self.load_models()

        # Read and preprocess
        data, meta = preprocessor.read_geotiff(image_path)
        if preprocessor.normalize:
            data = preprocessor.normalize_image(data)

        tiles = preprocessor.tile_image(data, meta)
        tile_data = [t["data"] for t in tiles]

        logger.info(f"Running segmentation on {len(tile_data)} tiles...")

        # Run model — returns class logits
        def seg_fn(batch):
            logits = self.seg_model(batch)
            return logits.argmax(dim=1)  # (B, H, W)

        predictions = self._run_batched(seg_fn, tile_data)

        # Stitch (class indices, so no averaging — use mode)
        class_map = preprocessor.stitch_tiles(tiles, predictions, meta, num_classes=1)

        # Round to nearest class
        class_map = np.round(class_map).astype(np.uint8)

        if output_path:
            preprocessor.save_geotiff(class_map, output_path, meta, dtype="uint8")

        logger.info("Segmentation complete")
        return class_map, meta

    def full_analysis(
        self,
        t1_path: Path,
        t2_path: Path,
        output_dir: Path,
    ) -> dict:
        """
        Run complete analysis: change detection + segmentation on T2.

        Returns:
            Dict with change_map, seg_map, and metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Change detection
        change_map, meta = self.detect_changes(
            t1_path, t2_path,
            output_path=output_dir / "change_map.tif",
        )

        # Segmentation on the after image
        seg_map, _ = self.segment_landcover(
            t2_path,
            output_path=output_dir / "segmentation_map.tif",
        )

        # Binary change mask
        change_mask = (change_map > self.confidence_threshold).astype(np.uint8)
        preprocessor.save_geotiff(
            change_mask, output_dir / "change_mask.tif", meta, dtype="uint8"
        )

        logger.info(f"Full analysis complete. Results in {output_dir}")

        return {
            "change_map": change_map,
            "change_mask": change_mask,
            "segmentation_map": seg_map,
            "metadata": meta,
        }


# Module-level singleton
pipeline = InferencePipeline()
