"""
Alert generation module.
Creates structured regulatory alerts from spatial analysis results
with severity scoring, geographic coordinates, and metadata.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
from loguru import logger


# Severity weights
SEVERITY_WEIGHTS = {
    "area": 0.3,         # Larger area = higher severity
    "sensitivity": 0.35,  # More sensitive zone = higher severity
    "confidence": 0.2,    # Model confidence
    "boundary": 0.15,     # Crossing cadastral boundaries
}

# Severity levels
SEVERITY_LEVELS = {
    (0.0, 0.3): "LOW",
    (0.3, 0.6): "MEDIUM",
    (0.6, 0.8): "HIGH",
    (0.8, 1.0): "CRITICAL",
}


def compute_severity_score(
    area_m2: float,
    zone_sensitivity: float,
    model_confidence: float,
    crosses_boundary: bool,
    max_area_m2: float = 10000.0,
) -> tuple[float, str]:
    """
    Compute a composite severity score for a detected change.

    Args:
        area_m2: Area of the change in square meters
        zone_sensitivity: Zone sensitivity (0-1)
        model_confidence: Model prediction confidence (0-1)
        crosses_boundary: Whether change crosses cadastral boundaries
        max_area_m2: Reference area for normalization

    Returns:
        Tuple of (score 0-1, severity level string)
    """
    # Normalize area (capped at max)
    area_score = min(area_m2 / max_area_m2, 1.0)

    # Boundary crossing is binary
    boundary_score = 1.0 if crosses_boundary else 0.0

    # Weighted combination
    score = (
        SEVERITY_WEIGHTS["area"] * area_score
        + SEVERITY_WEIGHTS["sensitivity"] * zone_sensitivity
        + SEVERITY_WEIGHTS["confidence"] * model_confidence
        + SEVERITY_WEIGHTS["boundary"] * boundary_score
    )

    score = min(max(score, 0.0), 1.0)

    # Map to level
    level = "LOW"
    for (lo, hi), lvl in SEVERITY_LEVELS.items():
        if lo <= score < hi:
            level = lvl
            break
    if score >= 0.8:
        level = "CRITICAL"

    return score, level


class AlertGenerator:
    """Generates structured alerts from analyzed change polygons."""

    def __init__(self):
        self.alerts: list[dict] = []

    def generate_alerts(
        self,
        analyzed_gdf: gpd.GeoDataFrame,
        t1_date: str = "unknown",
        t2_date: str = "unknown",
        analysis_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Generate alerts from analyzed GeoDataFrame.

        Args:
            analyzed_gdf: GeoDataFrame with columns from spatial_analysis
                          (zone_type, sensitivity, is_violation, area_m2, etc.)
            t1_date: Before-image date string
            t2_date: After-image date string
            analysis_id: Optional ID for this analysis run

        Returns:
            List of alert dicts
        """
        if analysis_id is None:
            analysis_id = str(uuid.uuid4())[:8]

        alerts = []
        violations = analyzed_gdf[
            analyzed_gdf.get("is_violation", False) == True
        ] if "is_violation" in analyzed_gdf.columns else analyzed_gdf

        for idx, row in violations.iterrows():
            geom = row.geometry
            centroid = geom.centroid

            # Get values with defaults
            area = row.get("area_m2", geom.area * 111000 * 111000)  # rough deg to m
            sensitivity = row.get("sensitivity", 0.5)
            confidence = row.get("change_value", 0.7)
            crosses = row.get("crosses_boundary", False) or False

            severity_score, severity_level = compute_severity_score(
                area_m2=area,
                zone_sensitivity=sensitivity,
                model_confidence=confidence,
                crosses_boundary=crosses,
            )

            alert = {
                "alert_id": f"{analysis_id}-{idx:04d}",
                "analysis_id": analysis_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity_score": round(severity_score, 3),
                "severity_level": severity_level,
                "coordinates": {
                    "latitude": round(centroid.y, 6),
                    "longitude": round(centroid.x, 6),
                },
                "bbox": {
                    "west": round(geom.bounds[0], 6),
                    "south": round(geom.bounds[1], 6),
                    "east": round(geom.bounds[2], 6),
                    "north": round(geom.bounds[3], 6),
                },
                "geometry": mapping(geom),
                "area_m2": round(area, 1),
                "zone_type": row.get("zone_type", "unknown"),
                "landuse": row.get("landuse", "unknown"),
                "violation_type": row.get("violation_type", "unauthorized_development"),
                "model_confidence": round(confidence, 3),
                "temporal_range": {
                    "before_date": t1_date,
                    "after_date": t2_date,
                },
                "cadastral": {
                    "within_parcel": row.get("within_parcel"),
                    "crosses_boundary": crosses,
                    "parcel_id": row.get("parcel_id"),
                },
                "status": "new",
            }

            alerts.append(alert)

        self.alerts.extend(alerts)

        # Summary
        level_counts = {}
        for a in alerts:
            level_counts[a["severity_level"]] = level_counts.get(a["severity_level"], 0) + 1

        logger.info(
            f"Generated {len(alerts)} alerts: {level_counts}"
        )

        return alerts

    def to_geojson(self, alerts: Optional[list[dict]] = None) -> dict:
        """Convert alerts to GeoJSON FeatureCollection for map display."""
        alerts = alerts or self.alerts

        features = []
        for alert in alerts:
            feature = {
                "type": "Feature",
                "geometry": alert["geometry"],
                "properties": {
                    k: v for k, v in alert.items()
                    if k != "geometry"
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def filter_alerts(
        self,
        min_severity: float = 0.0,
        severity_level: Optional[str] = None,
        zone_type: Optional[str] = None,
    ) -> list[dict]:
        """Filter stored alerts by criteria."""
        filtered = self.alerts

        if min_severity > 0:
            filtered = [a for a in filtered if a["severity_score"] >= min_severity]

        if severity_level:
            filtered = [a for a in filtered if a["severity_level"] == severity_level]

        if zone_type:
            filtered = [a for a in filtered if a["zone_type"] == zone_type]

        return filtered

    def get_summary(self) -> dict:
        """Get alert summary statistics."""
        if not self.alerts:
            return {"total": 0}

        scores = [a["severity_score"] for a in self.alerts]
        levels = [a["severity_level"] for a in self.alerts]

        return {
            "total": len(self.alerts),
            "avg_severity": round(np.mean(scores), 3),
            "max_severity": round(max(scores), 3),
            "by_level": {
                level: levels.count(level)
                for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                if levels.count(level) > 0
            },
            "by_zone": {
                zone: sum(1 for a in self.alerts if a["zone_type"] == zone)
                for zone in set(a["zone_type"] for a in self.alerts)
            },
        }


# Module-level singleton
alert_generator = AlertGenerator()
