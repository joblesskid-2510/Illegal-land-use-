"""
Sentinel-2 specific utilities: band helpers, spectral indices, and
temporal pair selection for change detection.
"""

from typing import Optional

import ee
import numpy as np
from loguru import logger


# Sentinel-2 band metadata
S2_BANDS = {
    "B2": {"name": "Blue", "wavelength": 490, "resolution": 10},
    "B3": {"name": "Green", "wavelength": 560, "resolution": 10},
    "B4": {"name": "Red", "wavelength": 665, "resolution": 10},
    "B5": {"name": "Red Edge 1", "wavelength": 705, "resolution": 20},
    "B6": {"name": "Red Edge 2", "wavelength": 740, "resolution": 20},
    "B7": {"name": "Red Edge 3", "wavelength": 783, "resolution": 20},
    "B8": {"name": "NIR", "wavelength": 842, "resolution": 10},
    "B8A": {"name": "NIR Narrow", "wavelength": 865, "resolution": 20},
    "B11": {"name": "SWIR 1", "wavelength": 1610, "resolution": 20},
    "B12": {"name": "SWIR 2", "wavelength": 2190, "resolution": 20},
}

# Preset band combinations
BAND_PRESETS = {
    "rgb": ["B4", "B3", "B2"],
    "false_color": ["B8", "B4", "B3"],
    "urban": ["B12", "B11", "B4"],
    "vegetation": ["B8", "B4", "B3"],
    "all_10m": ["B2", "B3", "B4", "B8"],
    "multispectral": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    "change_detection": ["B2", "B3", "B4", "B8", "B11", "B12"],
}


def compute_ndvi(image: ee.Image) -> ee.Image:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)"""
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def compute_ndbi(image: ee.Image) -> ee.Image:
    """Normalized Difference Built-up Index: (SWIR1 - NIR) / (SWIR1 + NIR)"""
    return image.normalizedDifference(["B11", "B8"]).rename("NDBI")


def compute_ndwi(image: ee.Image) -> ee.Image:
    """Normalized Difference Water Index: (Green - NIR) / (Green + NIR)"""
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


def compute_bsi(image: ee.Image) -> ee.Image:
    """
    Bare Soil Index:
    BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    """
    numerator = image.select("B11").add(image.select("B4")).subtract(
        image.select("B8").add(image.select("B2"))
    )
    denominator = image.select("B11").add(image.select("B4")).add(
        image.select("B8").add(image.select("B2"))
    )
    return numerator.divide(denominator).rename("BSI")


def compute_all_indices(image: ee.Image) -> ee.Image:
    """Append all spectral indices as additional bands."""
    ndvi = compute_ndvi(image)
    ndbi = compute_ndbi(image)
    ndwi = compute_ndwi(image)
    bsi = compute_bsi(image)
    return image.addBands([ndvi, ndbi, ndwi, bsi])


def select_best_temporal_pair(
    collection: ee.ImageCollection,
    min_gap_days: int = 90,
    max_gap_days: int = 365,
) -> tuple[ee.Image, ee.Image]:
    """
    Select the best before/after image pair from a collection
    for change detection based on minimal cloud cover
    and temporal separation.

    Args:
        collection: Filtered Sentinel-2 collection
        min_gap_days: Minimum days between T1 and T2
        max_gap_days: Maximum days between T1 and T2

    Returns:
        Tuple of (T1 image, T2 image)
    """
    sorted_col = collection.sort("system:time_start")
    img_list = sorted_col.toList(sorted_col.size())
    size = sorted_col.size().getInfo()

    if size < 2:
        raise ValueError(f"Need at least 2 images, got {size}")

    # Pick lowest-cloud image as T2 (most recent good image)
    best = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    t2_time = best.date().millis().getInfo()

    # Find T1: furthest back in time from T2 with acceptable gap
    candidates = collection.filterDate(
        ee.Date(t2_time).advance(-max_gap_days, "day"),
        ee.Date(t2_time).advance(-min_gap_days, "day"),
    ).sort("CLOUDY_PIXEL_PERCENTAGE")

    t1_count = candidates.size().getInfo()
    if t1_count == 0:
        logger.warning("No T1 candidates found, using earliest image in collection")
        t1 = ee.Image(img_list.get(0))
    else:
        t1 = candidates.first()

    t1_date = t1.date().format("YYYY-MM-dd").getInfo()
    t2_date = best.date().format("YYYY-MM-dd").getInfo()
    logger.info(f"Selected temporal pair: T1={t1_date}, T2={t2_date}")

    return t1, best


def normalize_to_uint8(array: np.ndarray, percentile: tuple = (2, 98)) -> np.ndarray:
    """
    Normalize a numpy array to 0-255 uint8 using percentile clipping.
    Useful for visualization of satellite bands.
    """
    p_low, p_high = np.percentile(array, percentile)
    clipped = np.clip(array, p_low, p_high)
    normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    return normalized
