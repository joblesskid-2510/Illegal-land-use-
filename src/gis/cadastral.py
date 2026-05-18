"""
Cadastral boundary checks.
Cross-references detected changes with property/parcel boundaries.
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
from loguru import logger

from config.settings import settings


class CadastralChecker:
    """
    Checks if detected land changes cross cadastral (property) boundaries
    or occur outside registered parcels.
    """

    def __init__(self, cadastral_data: Optional[gpd.GeoDataFrame] = None):
        self.cadastral_data = cadastral_data

    def load_cadastral_data(self, path: Path) -> gpd.GeoDataFrame:
        """Load cadastral/parcel boundary data from shapefile or GeoJSON."""
        self.cadastral_data = gpd.read_file(path)
        logger.info(f"Loaded {len(self.cadastral_data)} cadastral parcels from {path.name}")
        return self.cadastral_data

    def check_boundary_violations(
        self,
        change_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Check if changes cross parcel boundaries or are outside registered parcels.

        Adds columns:
        - within_parcel: bool — is the change fully inside a registered parcel
        - crosses_boundary: bool — does the change intersect parcel boundaries
        - parcel_id: ID of the containing parcel (if any)
        """
        if self.cadastral_data is None or len(self.cadastral_data) == 0:
            logger.warning("No cadastral data loaded. Skipping boundary checks.")
            change_gdf["within_parcel"] = None
            change_gdf["crosses_boundary"] = None
            change_gdf["parcel_id"] = None
            return change_gdf

        # Ensure CRS match
        if change_gdf.crs != self.cadastral_data.crs:
            change_gdf = change_gdf.to_crs(self.cadastral_data.crs)

        results = []
        for idx, change in change_gdf.iterrows():
            change_geom = change.geometry

            # Check containment
            containing = self.cadastral_data[
                self.cadastral_data.geometry.contains(change_geom)
            ]

            # Check intersection (crosses boundary)
            intersecting = self.cadastral_data[
                self.cadastral_data.geometry.intersects(change_geom)
                & ~self.cadastral_data.geometry.contains(change_geom)
            ]

            within = len(containing) > 0
            crosses = len(intersecting) > 0

            parcel_id = None
            if within and "parcel_id" in self.cadastral_data.columns:
                parcel_id = containing.iloc[0]["parcel_id"]

            results.append({
                "within_parcel": within,
                "crosses_boundary": crosses,
                "parcel_id": parcel_id,
                "num_parcels_affected": len(containing) + len(intersecting),
            })

        for col in ["within_parcel", "crosses_boundary", "parcel_id", "num_parcels_affected"]:
            change_gdf[col] = [r[col] for r in results]

        boundary_violations = change_gdf[change_gdf["crosses_boundary"] == True]
        outside_parcels = change_gdf[change_gdf["within_parcel"] == False]

        logger.info(
            f"Cadastral check: {len(boundary_violations)} boundary crossings, "
            f"{len(outside_parcels)} changes outside registered parcels"
        )

        return change_gdf


# Module-level singleton
cadastral_checker = CadastralChecker()
