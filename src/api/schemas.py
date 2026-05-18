"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────

class BBoxSchema(BaseModel):
    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)


class AnalysisRequest(BaseModel):
    """Request body for triggering an analysis."""
    bbox: Optional[BBoxSchema] = None  # Uses default AOI if None
    t1_start: str = Field(..., description="Start date for before period (YYYY-MM-DD)")
    t1_end: str = Field(..., description="End date for before period (YYYY-MM-DD)")
    t2_start: str = Field(..., description="Start date for after period (YYYY-MM-DD)")
    t2_end: str = Field(..., description="End date for after period (YYYY-MM-DD)")
    max_cloud_pct: float = Field(default=20.0, ge=0, le=100)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)


class AlertFilterParams(BaseModel):
    """Query parameters for filtering alerts."""
    min_severity: float = Field(default=0.0, ge=0, le=1)
    severity_level: Optional[str] = Field(
        default=None,
        pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$",
    )
    zone_type: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# ── Response Schemas ─────────────────────────────────────────────

class CoordinateSchema(BaseModel):
    latitude: float
    longitude: float


class TemporalRange(BaseModel):
    before_date: str
    after_date: str


class CadastralInfo(BaseModel):
    within_parcel: Optional[bool] = None
    crosses_boundary: Optional[bool] = None
    parcel_id: Optional[str] = None


class AlertResponse(BaseModel):
    alert_id: str
    analysis_id: str
    timestamp: str
    severity_score: float
    severity_level: str
    coordinates: CoordinateSchema
    bbox: BBoxSchema
    area_m2: float
    zone_type: str
    landuse: str
    violation_type: str
    model_confidence: float
    temporal_range: TemporalRange
    cadastral: CadastralInfo
    status: str


class AlertSummary(BaseModel):
    total: int
    avg_severity: Optional[float] = None
    max_severity: Optional[float] = None
    by_level: Optional[dict[str, int]] = None
    by_zone: Optional[dict[str, int]] = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    num_alerts: int = 0
    summary: Optional[AlertSummary] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    gee_authenticated: bool
    models_loaded: bool
