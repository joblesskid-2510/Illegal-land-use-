"""
Alert routes — CRUD operations and GeoJSON export.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from src.api.schemas import AlertResponse, AlertSummary
from src.gis.alert_generator import alert_generator

router = APIRouter(prefix="/alerts", tags=["alerts"])

DEMO_ALERTS = [
    {"alert_id": "BLR-0001", "analysis_id": "demo-run", "timestamp": "2025-05-18T00:00:00Z", "severity_score": 0.87, "severity_level": "CRITICAL", "coordinates": {"latitude": 12.9125, "longitude": 77.5225}, "bbox": {"west": 77.52, "south": 12.91, "east": 77.525, "north": 12.915}, "geometry": {"type": "Polygon", "coordinates": [[[77.52, 12.91], [77.525, 12.91], [77.525, 12.915], [77.52, 12.915], [77.52, 12.91]]]}, "area_m2": 2450.0, "zone_type": "agricultural", "landuse": "farmland", "violation_type": "unauthorized_development_in_agricultural", "model_confidence": 0.82, "temporal_range": {"before_date": "2024-01-15", "after_date": "2024-11-20"}, "cadastral": {"within_parcel": False, "crosses_boundary": True, "parcel_id": None}, "status": "new"},
    {"alert_id": "BLR-0002", "analysis_id": "demo-run", "timestamp": "2025-05-18T00:00:00Z", "severity_score": 0.72, "severity_level": "HIGH", "coordinates": {"latitude": 12.9725, "longitude": 77.6125}, "bbox": {"west": 77.61, "south": 12.97, "east": 77.618, "north": 12.978}, "geometry": {"type": "Polygon", "coordinates": [[[77.61, 12.97], [77.618, 12.97], [77.618, 12.978], [77.61, 12.978], [77.61, 12.97]]]}, "area_m2": 5200.0, "zone_type": "protected", "landuse": "forest", "violation_type": "unauthorized_development_in_protected", "model_confidence": 0.78, "temporal_range": {"before_date": "2024-02-01", "after_date": "2024-10-30"}, "cadastral": {"within_parcel": False, "crosses_boundary": False, "parcel_id": None}, "status": "new"},
    {"alert_id": "BLR-0003", "analysis_id": "demo-run", "timestamp": "2025-05-18T00:00:00Z", "severity_score": 0.55, "severity_level": "MEDIUM", "coordinates": {"latitude": 12.9315, "longitude": 77.5515}, "bbox": {"west": 77.55, "south": 12.93, "east": 77.556, "north": 12.936}, "geometry": {"type": "Polygon", "coordinates": [[[77.55, 12.93], [77.556, 12.93], [77.556, 12.936], [77.55, 12.936], [77.55, 12.93]]]}, "area_m2": 980.0, "zone_type": "green_space", "landuse": "grass", "violation_type": "unauthorized_development_in_green_space", "model_confidence": 0.65, "temporal_range": {"before_date": "2024-03-01", "after_date": "2024-12-01"}, "cadastral": {"within_parcel": True, "crosses_boundary": False, "parcel_id": "BLR-4521"}, "status": "new"},
    {"alert_id": "BLR-0004", "analysis_id": "demo-run", "timestamp": "2025-05-18T00:00:00Z", "severity_score": 0.92, "severity_level": "CRITICAL", "coordinates": {"latitude": 12.895, "longitude": 77.685}, "bbox": {"west": 77.682, "south": 12.892, "east": 77.690, "north": 12.900}, "geometry": {"type": "Polygon", "coordinates": [[[77.682, 12.892], [77.690, 12.892], [77.690, 12.900], [77.682, 12.900], [77.682, 12.892]]]}, "area_m2": 7800.0, "zone_type": "water", "landuse": "reservoir", "violation_type": "unauthorized_development_in_water", "model_confidence": 0.91, "temporal_range": {"before_date": "2024-01-10", "after_date": "2024-09-15"}, "cadastral": {"within_parcel": False, "crosses_boundary": True, "parcel_id": None}, "status": "new"},
    {"alert_id": "BLR-0005", "analysis_id": "demo-run", "timestamp": "2025-05-18T00:00:00Z", "severity_score": 0.68, "severity_level": "HIGH", "coordinates": {"latitude": 12.945, "longitude": 77.635}, "bbox": {"west": 77.632, "south": 12.942, "east": 77.639, "north": 12.949}, "geometry": {"type": "Polygon", "coordinates": [[[77.632, 12.942], [77.639, 12.942], [77.639, 12.949], [77.632, 12.949], [77.632, 12.942]]]}, "area_m2": 3400.0, "zone_type": "agricultural", "landuse": "farmland", "violation_type": "unauthorized_development_in_agricultural", "model_confidence": 0.73, "temporal_range": {"before_date": "2024-04-01", "after_date": "2024-11-30"}, "cadastral": {"within_parcel": False, "crosses_boundary": True, "parcel_id": None}, "status": "new"},
    {"alert_id": "BLR-0006", "analysis_id": "demo-run", "timestamp": "2025-05-18T00:00:00Z", "severity_score": 0.22, "severity_level": "LOW", "coordinates": {"latitude": 12.960, "longitude": 77.580}, "bbox": {"west": 77.578, "south": 12.958, "east": 77.582, "north": 12.962}, "geometry": {"type": "Polygon", "coordinates": [[[77.578, 12.958], [77.582, 12.958], [77.582, 12.962], [77.578, 12.962], [77.578, 12.958]]]}, "area_m2": 420.0, "zone_type": "residential", "landuse": "residential", "violation_type": "permitted_zone", "model_confidence": 0.55, "temporal_range": {"before_date": "2024-05-01", "after_date": "2024-12-15"}, "cadastral": {"within_parcel": True, "crosses_boundary": False, "parcel_id": "BLR-7783"}, "status": "new"},
]


@router.on_event("startup")
async def seed_demo_data():
    """Auto-seed demo alerts on startup."""
    if not alert_generator.alerts:
        alert_generator.alerts.extend(DEMO_ALERTS)
        logger.info(f"Seeded {len(DEMO_ALERTS)} demo alerts")


@router.get("/", response_model=list[AlertResponse])
async def list_alerts(
    min_severity: float = Query(default=0.0, ge=0, le=1),
    severity_level: Optional[str] = Query(
        default=None,
        pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$",
    ),
    zone_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List alerts with optional filters."""
    filtered = alert_generator.filter_alerts(
        min_severity=min_severity,
        severity_level=severity_level,
        zone_type=zone_type,
    )

    # Pagination
    paginated = filtered[offset : offset + limit]

    return paginated


@router.get("/summary", response_model=AlertSummary)
async def get_alert_summary():
    """Get summary statistics of all alerts."""
    return alert_generator.get_summary()


@router.get("/geojson")
async def get_alerts_geojson(
    min_severity: float = Query(default=0.0, ge=0, le=1),
    severity_level: Optional[str] = Query(default=None),
):
    """Get alerts as GeoJSON FeatureCollection for map display."""
    filtered = alert_generator.filter_alerts(
        min_severity=min_severity,
        severity_level=severity_level,
    )
    return alert_generator.to_geojson(filtered)


@router.get("/{alert_id}")
async def get_alert(alert_id: str):
    """Get a single alert by ID."""
    for alert in alert_generator.alerts:
        if alert["alert_id"] == alert_id:
            return alert

    raise HTTPException(status_code=404, detail="Alert not found")


@router.patch("/{alert_id}/status")
async def update_alert_status(alert_id: str, status: str = Query(...)):
    """Update the status of an alert (new, reviewed, resolved, false_positive)."""
    valid_statuses = ["new", "reviewed", "resolved", "false_positive"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}",
        )

    for alert in alert_generator.alerts:
        if alert["alert_id"] == alert_id:
            alert["status"] = status
            logger.info(f"Alert {alert_id} status updated to {status}")
            return {"message": f"Status updated to {status}", "alert_id": alert_id}

    raise HTTPException(status_code=404, detail="Alert not found")
