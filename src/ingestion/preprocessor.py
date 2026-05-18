"""
Image preprocessing pipeline: tiling, normalization, co-registration,
and conversion to model-ready format.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from loguru import logger

from config.settings import settings


class ImagePreprocessor:
    """Preprocesses satellite GeoTIFFs into tiles for model inference."""

    def __init__(
        self,
        tile_size: int = 256,
        tile_overlap: int = 32,
        normalize: bool = True,
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.normalize = normalize

    def read_geotiff(self, path: Path) -> tuple[np.ndarray, dict]:
        """
        Read a GeoTIFF and return array + metadata.

        Returns:
            Tuple of (array [C, H, W], metadata dict with crs, transform, etc.)
        """
        with rasterio.open(path) as src:
            data = src.read()  # shape: (bands, height, width)
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "bounds": src.bounds,
                "nodata": src.nodata,
            }
        logger.debug(f"Read {path.name}: shape={data.shape}, crs={meta['crs']}")
        return data, meta

    def normalize_image(self, data: np.ndarray) -> np.ndarray:
        """
        Per-band min-max normalization to [0, 1].
        Handles nodata by masking zeros.
        """
        normalized = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            band = data[i].astype(np.float32)
            valid = band[band > 0]
            if len(valid) == 0:
                continue
            bmin, bmax = np.percentile(valid, [2, 98])
            if bmax - bmin > 0:
                normalized[i] = np.clip((band - bmin) / (bmax - bmin), 0, 1)
        return normalized

    def tile_image(
        self,
        data: np.ndarray,
        meta: dict,
    ) -> list[dict]:
        """
        Split image into overlapping tiles.

        Returns:
            List of dicts with 'data' (C, H, W), 'row', 'col', 'window', 'transform'
        """
        _, height, width = data.shape
        step = self.tile_size - self.tile_overlap
        tiles = []

        for row in range(0, height - self.tile_size + 1, step):
            for col in range(0, width - self.tile_size + 1, step):
                window = Window(col, row, self.tile_size, self.tile_size)
                tile_data = data[:, row:row + self.tile_size, col:col + self.tile_size]

                # Skip tiles that are mostly nodata (>50% zeros)
                if np.mean(tile_data == 0) > 0.5:
                    continue

                tile_transform = rasterio.windows.transform(window, meta["transform"])

                tiles.append({
                    "data": tile_data,
                    "row": row,
                    "col": col,
                    "window": window,
                    "transform": tile_transform,
                })

        logger.info(f"Generated {len(tiles)} tiles ({self.tile_size}x{self.tile_size}, overlap={self.tile_overlap})")
        return tiles

    def prepare_pair(
        self,
        t1_path: Path,
        t2_path: Path,
        output_dir: Optional[Path] = None,
    ) -> tuple[list[dict], list[dict], dict]:
        """
        Full preprocessing pipeline for a bi-temporal image pair.

        Args:
            t1_path: Path to before-image GeoTIFF
            t2_path: Path to after-image GeoTIFF
            output_dir: Optional directory to save preprocessed tiles as .npy

        Returns:
            Tuple of (t1_tiles, t2_tiles, metadata)
        """
        # Read
        t1_data, t1_meta = self.read_geotiff(t1_path)
        t2_data, t2_meta = self.read_geotiff(t2_path)

        # Validate shapes match
        if t1_data.shape != t2_data.shape:
            raise ValueError(
                f"Shape mismatch: T1={t1_data.shape} vs T2={t2_data.shape}. "
                "Images must be co-registered and same resolution."
            )

        # Normalize
        if self.normalize:
            t1_data = self.normalize_image(t1_data)
            t2_data = self.normalize_image(t2_data)

        # Tile
        t1_tiles = self.tile_image(t1_data, t1_meta)
        t2_tiles = self.tile_image(t2_data, t2_meta)

        # Save tiles if output dir specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for i, (tile1, tile2) in enumerate(zip(t1_tiles, t2_tiles)):
                np.save(output_dir / f"t1_tile_{i:04d}.npy", tile1["data"])
                np.save(output_dir / f"t2_tile_{i:04d}.npy", tile2["data"])
            logger.info(f"Saved {len(t1_tiles)} tile pairs to {output_dir}")

        return t1_tiles, t2_tiles, t1_meta

    def stitch_tiles(
        self,
        tiles: list[dict],
        predictions: list[np.ndarray],
        original_meta: dict,
        num_classes: int = 1,
    ) -> np.ndarray:
        """
        Stitch tile predictions back into a full-scene array.
        Uses averaging for overlapping regions.

        Args:
            tiles: Original tile dicts (with row, col info)
            predictions: List of prediction arrays (H, W) or (C, H, W)
            original_meta: Metadata from original image
            num_classes: Number of output channels

        Returns:
            Full-scene prediction array
        """
        height = original_meta["height"]
        width = original_meta["width"]

        if num_classes == 1:
            output = np.zeros((height, width), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.float32)
        else:
            output = np.zeros((num_classes, height, width), dtype=np.float32)
            counts = np.zeros((1, height, width), dtype=np.float32)

        for tile, pred in zip(tiles, predictions):
            r, c = tile["row"], tile["col"]
            ts = self.tile_size

            if num_classes == 1:
                output[r:r + ts, c:c + ts] += pred
                counts[r:r + ts, c:c + ts] += 1
            else:
                output[:, r:r + ts, c:c + ts] += pred
                counts[:, r:r + ts, c:c + ts] += 1

        # Average overlapping regions
        counts = np.maximum(counts, 1)
        output /= counts

        return output

    def save_geotiff(
        self,
        data: np.ndarray,
        output_path: Path,
        meta: dict,
        dtype: str = "float32",
    ) -> Path:
        """Save a numpy array as a GeoTIFF with original georeferencing."""
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # Add channel dim

        count = data.shape[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=count,
            dtype=dtype,
            crs=meta["crs"],
            transform=meta["transform"],
            compress="lzw",
        ) as dst:
            for i in range(count):
                dst.write(data[i], i + 1)

        logger.info(f"Saved GeoTIFF: {output_path}")
        return output_path


# Module-level instance
preprocessor = ImagePreprocessor(
    tile_size=settings.model.tile_size,
    tile_overlap=settings.model.tile_overlap,
)
