"""
Spatial analysis module.
Converts raster change/segmentation results to vector features,
overlays with zoning/land-use boundaries, and identifies violations.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from loguru import logger

from config.settings import settings


class SpatialAnalyzer:
    """
    Performs GIS spatial analysis:
    - Raster to vector polygon conversion
    - Zoning overlay
    - Violation identification
    """

    def __init__(self, zoning_data: Optional[gpd.GeoDataFrame] = None):
        self.zoning_data = zoning_data

    def load_zoning_from_file(self, path: Path) -> gpd.GeoDataFrame:
        """Load zoning/land-use shapefile or GeoJSON."""
        self.zoning_data = gpd.read_file(path)
        logger.info(f"Loaded zoning data: {len(self.zoning_data)} features from {path.name}")
        return self.zoning_data

    def load_zoning_from_osm(self, bbox: list[float]) -> gpd.GeoDataFrame:
        """
        Fetch land-use data from OpenStreetMap as a zoning proxy.
        bbox: [west, south, east, north]
        """
        import httpx

        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:60];
        (
          way["landuse"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          relation["landuse"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = httpx.post(overpass_url, data={"data": query}, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Parse OSM elements to GeoDataFrame
            features = self._parse_osm_landuse(data)
            if features:
                self.zoning_data = gpd.GeoDataFrame(features, crs="EPSG:4326")
                logger.info(f"Fetched {len(self.zoning_data)} land-use polygons from OSM")
            else:
                logger.warning("No land-use features found in OSM for this AOI")
                self.zoning_data = gpd.GeoDataFrame(
                    columns=["geometry", "landuse", "zone_type", "sensitivity"],
                    crs="EPSG:4326",
                )
        except Exception as e:
            logger.error(f"Failed to fetch OSM data: {e}")
            self.zoning_data = gpd.GeoDataFrame(
                columns=["geometry", "landuse", "zone_type", "sensitivity"],
                crs="EPSG:4326",
            )

        return self.zoning_data

    def _parse_osm_landuse(self, osm_data: dict) -> list[dict]:
        """Parse Overpass API response into feature dicts."""
        from shapely.geometry import Polygon

        nodes = {}
        ways = {}
        features = []

        for element in osm_data.get("elements", []):
            if element["type"] == "node":
                nodes[element["id"]] = (element["lon"], element["lat"])
            elif element["type"] == "way":
                ways[element["id"]] = element

        # Sensitivity mapping for different land-use types
        sensitivity_map = {
            "residential": ("residential", 0.7),
            "commercial": ("commercial", 0.5),
            "industrial": ("industrial", 0.4),
            "farmland": ("agricultural", 0.9),
            "forest": ("protected", 1.0),
            "meadow": ("agricultural", 0.8),
            "grass": ("green_space", 0.6),
            "recreation_ground": ("recreational", 0.6),
            "military": ("restricted", 1.0),
            "cemetery": ("protected", 0.8),
            "reservoir": ("water", 0.9),
            "basin": ("water", 0.9),
            "conservation": ("protected", 1.0),
        }

        for way_id, way in ways.items():
            node_ids = way.get("nodes", [])
            coords = [nodes[nid] for nid in node_ids if nid in nodes]

            if len(coords) < 4:
                continue

            # Ensure closed ring
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                poly = Polygon(coords)
                if not poly.is_valid or poly.area == 0:
                    continue
            except Exception:
                continue

            landuse = way.get("tags", {}).get("landuse", "unknown")
            zone_type, sensitivity = sensitivity_map.get(landuse, ("other", 0.3))

            features.append({
                "geometry": poly,
                "landuse": landuse,
                "zone_type": zone_type,
                "sensitivity": sensitivity,
                "osm_id": way_id,
            })

        return features

    def raster_to_polygons(
        self,
        raster_path: Path,
        threshold: float = 0.5,
        min_area_m2: float = 100.0,
    ) -> gpd.GeoDataFrame:
        """
        Convert a change detection raster to vector polygons.

        Args:
            raster_path: Path to change map GeoTIFF
            threshold: Binary threshold for change
            min_area_m2: Minimum polygon area to keep

        Returns:
            GeoDataFrame of change polygons
        """
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Threshold to binary
        binary = (data > threshold).astype(np.uint8)

        # Extract polygon geometries
        results = []
        for geom, value in shapes(binary, transform=transform):
            if value == 1:
                poly = shape(geom)
                if poly.is_valid and poly.area > 0:
                    results.append({
                        "geometry": poly,
                        "change_value": float(value),
                    })

        gdf = gpd.GeoDataFrame(results, crs=crs)

        # Filter by area (approximate — proper area calc needs projection)
        if len(gdf) > 0:
            # Project to UTM for area calculation
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf["area_m2"] = gdf_utm.geometry.area
            gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)

        logger.info(f"Extracted {len(gdf)} change polygons (min area={min_area_m2}m²)")
        return gdf

    def overlay_with_zoning(
        self,
        change_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Spatial join of change polygons with zoning data.
        Identifies which zone each change falls in.
        """
        if self.zoning_data is None or len(self.zoning_data) == 0:
            logger.warning("No zoning data available. Loading from OSM...")
            self.load_zoning_from_osm(settings.aoi.bbox)

        if len(self.zoning_data) == 0:
            # No zoning data — mark all as "unzoned"
            change_gdf["zone_type"] = "unzoned"
            change_gdf["landuse"] = "unknown"
            change_gdf["sensitivity"] = 0.5
            return change_gdf

        # Ensure same CRS
        if change_gdf.crs != self.zoning_data.crs:
            change_gdf = change_gdf.to_crs(self.zoning_data.crs)

        # Spatial join
        joined = gpd.sjoin(
            change_gdf,
            self.zoning_data[["geometry", "zone_type", "landuse", "sensitivity"]],
            how="left",
            predicate="intersects",
        )

        # Fill non-overlapping as unzoned
        joined["zone_type"] = joined["zone_type"].fillna("unzoned")
        joined["landuse"] = joined["landuse"].fillna("unknown")
        joined["sensitivity"] = joined["sensitivity"].fillna(0.5)

        # Drop duplicate index columns from sjoin
        joined = joined.drop(columns=["index_right"], errors="ignore")

        # Deduplicate (a change polygon may intersect multiple zones — keep highest sensitivity)
        joined = (
            joined.sort_values("sensitivity", ascending=False)
            .drop_duplicates(subset=["geometry"], keep="first")
            .reset_index(drop=True)
        )

        logger.info(
            f"Overlay complete: {len(joined)} change features, "
            f"zones: {joined['zone_type'].value_counts().to_dict()}"
        )
        return joined

    def identify_violations(
        self,
        overlaid_gdf: gpd.GeoDataFrame,
        restricted_zones: Optional[list[str]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Flag change polygons that fall in restricted or sensitive zones.

        Args:
            overlaid_gdf: Result from overlay_with_zoning
            restricted_zones: Zone types considered illegal for development.
                Defaults to agricultural, protected, water, restricted.
        """
        if restricted_zones is None:
            restricted_zones = [
                "agricultural", "protected", "water", "restricted",
                "green_space", "recreational",
            ]

        overlaid_gdf["is_violation"] = overlaid_gdf["zone_type"].isin(restricted_zones)
        overlaid_gdf["violation_type"] = overlaid_gdf.apply(
            lambda row: f"unauthorized_development_in_{row['zone_type']}"
            if row["is_violation"] else "permitted_zone",
            axis=1,
        )

        violations = overlaid_gdf[overlaid_gdf["is_violation"]]
        logger.info(
            f"Identified {len(violations)} potential violations "
            f"out of {len(overlaid_gdf)} changes"
        )

        return overlaid_gdf


# Module-level singleton
spatial_analyzer = SpatialAnalyzer()
