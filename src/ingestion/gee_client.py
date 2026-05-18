"""
Google Earth Engine client for satellite imagery ingestion.
Handles authentication, image collection queries, and export to GeoTIFF.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import ee
from loguru import logger

from config.settings import settings


class GEEClient:
    """Client for Google Earth Engine API interactions."""

    def __init__(self):
        self._initialized = False

    def authenticate(self) -> None:
        """Authenticate with GEE using service account credentials."""
        key_path = settings.gee.key_path
        if not key_path.exists():
            raise FileNotFoundError(
                f"GEE service account key not found: {key_path}"
            )

        with open(key_path) as f:
            key_data = json.load(f)

        credentials = ee.ServiceAccountCredentials(
            email=key_data["client_email"],
            key_file=str(key_path),
        )
        ee.Initialize(
            credentials=credentials,
            project=settings.gee.project_id,
        )
        self._initialized = True
        logger.info("GEE authenticated successfully")

    def _ensure_init(self):
        if not self._initialized:
            self.authenticate()

    def get_sentinel2_collection(
        self,
        bbox: list[float],
        start_date: str,
        end_date: str,
        max_cloud_pct: float = 20.0,
    ) -> ee.ImageCollection:
        """
        Fetch Sentinel-2 L2A surface reflectance collection.

        Args:
            bbox: [west, south, east, north] in EPSG:4326
            start_date: ISO date string (YYYY-MM-DD)
            end_date: ISO date string (YYYY-MM-DD)
            max_cloud_pct: Maximum cloud cover percentage filter
        """
        self._ensure_init()

        aoi = ee.Geometry.Rectangle(bbox)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        count = collection.size().getInfo()
        logger.info(
            f"Found {count} Sentinel-2 images for "
            f"{start_date} to {end_date} (cloud ≤ {max_cloud_pct}%)"
        )
        return collection

    def get_landsat8_collection(
        self,
        bbox: list[float],
        start_date: str,
        end_date: str,
        max_cloud_pct: float = 20.0,
    ) -> ee.ImageCollection:
        """Fetch Landsat-8 Collection 2 Level 2 surface reflectance."""
        self._ensure_init()

        aoi = ee.Geometry.Rectangle(bbox)

        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUD_COVER", max_cloud_pct))
            .sort("CLOUD_COVER")
        )

        count = collection.size().getInfo()
        logger.info(f"Found {count} Landsat-8 images")
        return collection

    @staticmethod
    def mask_s2_clouds(image: ee.Image) -> ee.Image:
        """Apply cloud masking using Sentinel-2 QA60 band."""
        qa = image.select("QA60")
        # Bits 10 and 11 are clouds and cirrus
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)

    def get_composite(
        self,
        bbox: list[float],
        start_date: str,
        end_date: str,
        bands: Optional[list[str]] = None,
        max_cloud_pct: float = 20.0,
    ) -> ee.Image:
        """
        Create a cloud-free median composite from Sentinel-2.

        Args:
            bands: Band names to select. Defaults to multispectral bands.
        """
        if bands is None:
            bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

        collection = self.get_sentinel2_collection(
            bbox, start_date, end_date, max_cloud_pct
        )
        composite = (
            collection
            .map(self.mask_s2_clouds)
            .select(bands)
            .median()
            .clip(ee.Geometry.Rectangle(bbox))
        )
        return composite

    def get_temporal_pair(
        self,
        bbox: list[float],
        t1_start: str,
        t1_end: str,
        t2_start: str,
        t2_end: str,
        bands: Optional[list[str]] = None,
    ) -> tuple[ee.Image, ee.Image]:
        """
        Get a pair of composites for bi-temporal change detection.

        Returns:
            Tuple of (before_composite, after_composite)
        """
        before = self.get_composite(bbox, t1_start, t1_end, bands)
        after = self.get_composite(bbox, t2_start, t2_end, bands)
        logger.info(f"Created temporal pair: T1=[{t1_start},{t1_end}] T2=[{t2_start},{t2_end}]")
        return before, after

    def export_to_drive(
        self,
        image: ee.Image,
        description: str,
        folder: str = "landwatch_exports",
        scale: int = 10,
        crs: str = "EPSG:4326",
        bbox: Optional[list[float]] = None,
    ) -> ee.batch.Task:
        """Export an image to Google Drive as GeoTIFF."""
        self._ensure_init()

        region = ee.Geometry.Rectangle(bbox or settings.aoi.bbox)

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            scale=scale,
            region=region,
            crs=crs,
            fileFormat="GeoTIFF",
            maxPixels=1e10,
        )
        task.start()
        logger.info(f"Export task started: {description}")
        return task

    def export_to_local(
        self,
        image: ee.Image,
        output_path: Path,
        bbox: Optional[list[float]] = None,
        scale: int = 10,
        bands: Optional[list[str]] = None,
    ) -> Path:
        """
        Download image directly to local filesystem via getDownloadURL.
        For small AOIs only (< ~50 MB).
        """
        import httpx

        self._ensure_init()
        region = ee.Geometry.Rectangle(bbox or settings.aoi.bbox)

        params = {
            "scale": scale,
            "crs": "EPSG:4326",
            "region": region.getInfo()["coordinates"],
            "format": "GEO_TIFF",
        }
        if bands:
            params["bands"] = bands

        url = image.getDownloadURL(params)
        logger.info(f"Downloading image to {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=300) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"Downloaded: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path


# Module-level singleton
gee_client = GEEClient()
