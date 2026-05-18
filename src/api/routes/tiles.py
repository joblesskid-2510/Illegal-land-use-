"""
Tile serving routes — serve raster change/segmentation maps as slippy map tiles.
Uses rasterio to read GeoTIFFs and render PNG tiles on demand.
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from loguru import logger

from config.settings import settings

router = APIRouter(prefix="/tiles", tags=["tiles"])

# Color maps for different layer types
CHANGE_COLORMAP = {
    0: [0, 0, 0, 0],       # No change — transparent
    1: [255, 69, 0, 180],   # Change — red-orange, semi-transparent
}

SEGMENTATION_COLORMAP = {
    0: [0, 0, 0, 0],         # Background — transparent
    1: [255, 0, 0, 160],     # Building — red
    2: [128, 128, 128, 160], # Road — gray
    3: [0, 200, 0, 120],     # Vegetation — green
    4: [0, 100, 255, 160],   # Water — blue
    5: [210, 180, 140, 140], # Bare soil — tan
    6: [255, 165, 0, 180],   # Construction — orange
}


def _get_tile_from_raster(
    raster_path: Path,
    z: int,
    x: int,
    y: int,
    colormap: dict,
    tile_size: int = 256,
) -> Optional[bytes]:
    """
    Extract and render a map tile from a GeoTIFF.
    Uses web mercator tile coordinates (z/x/y).
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from PIL import Image

    if not raster_path.exists():
        return None

    # Convert tile coords to bounds (EPSG:4326)
    n = 2 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    import math
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    try:
        with rasterio.open(raster_path) as src:
            # Transform tile bounds to raster CRS
            src_bounds = transform_bounds(
                "EPSG:4326", src.crs,
                lon_min, lat_min, lon_max, lat_max,
            )

            # Check if tile overlaps raster
            raster_bounds = src.bounds
            if (
                src_bounds[0] > raster_bounds[2]
                or src_bounds[2] < raster_bounds[0]
                or src_bounds[1] > raster_bounds[3]
                or src_bounds[3] < raster_bounds[1]
            ):
                return None

            # Read data for this tile extent
            from rasterio.windows import from_bounds
            window = from_bounds(*src_bounds, transform=src.transform)

            data = src.read(
                1,
                window=window,
                out_shape=(tile_size, tile_size),
                resampling=rasterio.enums.Resampling.nearest,
            )

    except Exception as e:
        logger.debug(f"Tile {z}/{x}/{y} read error: {e}")
        return None

    # Apply colormap
    rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    for value, color in colormap.items():
        mask = (data == value) if isinstance(value, int) else (data > 0.5)
        rgba[mask] = color

    # If raster is float (change probability), use gradient
    if data.dtype in [np.float32, np.float64]:
        # Red channel intensity based on probability
        intensity = np.clip(data * 255, 0, 255).astype(np.uint8)
        rgba[:, :, 0] = intensity  # Red
        rgba[:, :, 1] = np.clip(255 - intensity, 50, 100).astype(np.uint8)  # Some green
        rgba[:, :, 2] = 0  # No blue
        rgba[:, :, 3] = np.where(data > 0.1, 180, 0).astype(np.uint8)  # Alpha

    # Convert to PNG
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@router.get("/change/{analysis_id}/{z}/{x}/{y}.png")
async def get_change_tile(analysis_id: str, z: int, x: int, y: int):
    """Serve change detection overlay tiles."""
    raster_path = settings.processed_dir / analysis_id / "change_map.tif"

    tile_data = _get_tile_from_raster(
        raster_path, z, x, y, CHANGE_COLORMAP
    )

    if tile_data is None:
        # Return transparent tile
        from PIL import Image
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tile_data = buf.getvalue()

    return Response(content=tile_data, media_type="image/png")


@router.get("/segmentation/{analysis_id}/{z}/{x}/{y}.png")
async def get_segmentation_tile(analysis_id: str, z: int, x: int, y: int):
    """Serve land cover segmentation overlay tiles."""
    raster_path = settings.processed_dir / analysis_id / "segmentation_map.tif"

    tile_data = _get_tile_from_raster(
        raster_path, z, x, y, SEGMENTATION_COLORMAP
    )

    if tile_data is None:
        from PIL import Image
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tile_data = buf.getvalue()

    return Response(content=tile_data, media_type="image/png")


@router.get("/layers")
async def list_available_layers():
    """List available tile layers."""
    layers = []

    processed = settings.processed_dir
    if processed.exists():
        for analysis_dir in processed.iterdir():
            if analysis_dir.is_dir():
                aid = analysis_dir.name
                has_change = (analysis_dir / "change_map.tif").exists()
                has_seg = (analysis_dir / "segmentation_map.tif").exists()

                if has_change:
                    layers.append({
                        "id": f"change_{aid}",
                        "type": "change_detection",
                        "analysis_id": aid,
                        "url": f"/tiles/change/{aid}/{{z}}/{{x}}/{{y}}.png",
                    })
                if has_seg:
                    layers.append({
                        "id": f"seg_{aid}",
                        "type": "segmentation",
                        "analysis_id": aid,
                        "url": f"/tiles/segmentation/{aid}/{{z}}/{{x}}/{{y}}.png",
                    })

    return {"layers": layers}
