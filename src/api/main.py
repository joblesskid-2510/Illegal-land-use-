"""
FastAPI application entry point.
Serves the API backend and the Leaflet.js dashboard.
"""

import sys
from pathlib import Path

# Ensure project root is in path BEFORE any project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from config.settings import settings

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

# Create FastAPI app
app = FastAPI(
    title="LandWatch AI",
    description="Satellite-Driven AI System for Detecting Unauthorized Land Development",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routes (graceful if heavy deps missing)
_alerts_loaded = False
_analysis_loaded = False

try:
    from src.api.routes.alerts import router as alerts_router
    app.include_router(alerts_router, prefix="/api")
    _alerts_loaded = True
except Exception as e:
    logger.warning(f"Full alerts routes unavailable ({e}), using built-in fallback")

try:
    from src.api.routes.analysis import router as analysis_router
    app.include_router(analysis_router, prefix="/api")
    _analysis_loaded = True
except Exception as e:
    logger.warning(f"Analysis routes unavailable ({e}), using built-in fallback")

try:
    from src.api.routes.tiles import router as tiles_router
    app.include_router(tiles_router, prefix="/api")
except Exception as e:
    logger.warning(f"Tiles routes unavailable: {e}")

# ── Fallback demo data (no heavy deps needed) ────────────────────

DEMO_ALERTS = [
    {"alert_id":"WRD-0001","analysis_id":"demo","timestamp":"2025-05-18T00:00:00Z","severity_score":0.87,"severity_level":"CRITICAL","coordinates":{"latitude":20.745,"longitude":78.600},"bbox":{"west":78.595,"south":20.740,"east":78.605,"north":20.750},"geometry":{"type":"Polygon","coordinates":[[[78.595,20.740],[78.605,20.740],[78.605,20.750],[78.595,20.750],[78.595,20.740]]]},"area_m2":2450.0,"zone_type":"agricultural","landuse":"farmland","violation_type":"unauthorized_development_in_agricultural","model_confidence":0.82,"temporal_range":{"before_date":"2024-01-15","after_date":"2024-11-20"},"cadastral":{"within_parcel":False,"crosses_boundary":True,"parcel_id":None},"status":"new"},
    {"alert_id":"WRD-0002","analysis_id":"demo","timestamp":"2025-05-18T00:00:00Z","severity_score":0.72,"severity_level":"HIGH","coordinates":{"latitude":20.780,"longitude":78.620},"bbox":{"west":78.616,"south":20.776,"east":78.624,"north":20.784},"geometry":{"type":"Polygon","coordinates":[[[78.616,20.776],[78.624,20.776],[78.624,20.784],[78.616,20.784],[78.616,20.776]]]},"area_m2":5200.0,"zone_type":"protected","landuse":"forest","violation_type":"unauthorized_development_in_protected","model_confidence":0.78,"temporal_range":{"before_date":"2024-02-01","after_date":"2024-10-30"},"cadastral":{"within_parcel":False,"crosses_boundary":False,"parcel_id":None},"status":"new"},
    {"alert_id":"WRD-0003","analysis_id":"demo","timestamp":"2025-05-18T00:00:00Z","severity_score":0.55,"severity_level":"MEDIUM","coordinates":{"latitude":20.730,"longitude":78.570},"bbox":{"west":78.567,"south":20.727,"east":78.573,"north":20.733},"geometry":{"type":"Polygon","coordinates":[[[78.567,20.727],[78.573,20.727],[78.573,20.733],[78.567,20.733],[78.567,20.727]]]},"area_m2":980.0,"zone_type":"green_space","landuse":"grass","violation_type":"unauthorized_development_in_green_space","model_confidence":0.65,"temporal_range":{"before_date":"2024-03-01","after_date":"2024-12-01"},"cadastral":{"within_parcel":True,"crosses_boundary":False,"parcel_id":"WRD-4521"},"status":"new"},
    {"alert_id":"WRD-0004","analysis_id":"demo","timestamp":"2025-05-18T00:00:00Z","severity_score":0.92,"severity_level":"CRITICAL","coordinates":{"latitude":20.700,"longitude":78.650},"bbox":{"west":78.646,"south":20.696,"east":78.654,"north":20.704},"geometry":{"type":"Polygon","coordinates":[[[78.646,20.696],[78.654,20.696],[78.654,20.704],[78.646,20.704],[78.646,20.696]]]},"area_m2":7800.0,"zone_type":"water","landuse":"reservoir","violation_type":"unauthorized_development_in_water","model_confidence":0.91,"temporal_range":{"before_date":"2024-01-10","after_date":"2024-09-15"},"cadastral":{"within_parcel":False,"crosses_boundary":True,"parcel_id":None},"status":"new"},
    {"alert_id":"WRD-0005","analysis_id":"demo","timestamp":"2025-05-18T00:00:00Z","severity_score":0.68,"severity_level":"HIGH","coordinates":{"latitude":20.760,"longitude":78.550},"bbox":{"west":78.546,"south":20.756,"east":78.554,"north":20.764},"geometry":{"type":"Polygon","coordinates":[[[78.546,20.756],[78.554,20.756],[78.554,20.764],[78.546,20.764],[78.546,20.756]]]},"area_m2":3400.0,"zone_type":"agricultural","landuse":"farmland","violation_type":"unauthorized_development_in_agricultural","model_confidence":0.73,"temporal_range":{"before_date":"2024-04-01","after_date":"2024-11-30"},"cadastral":{"within_parcel":False,"crosses_boundary":True,"parcel_id":None},"status":"new"},
    {"alert_id":"WRD-0006","analysis_id":"demo","timestamp":"2025-05-18T00:00:00Z","severity_score":0.22,"severity_level":"LOW","coordinates":{"latitude":20.750,"longitude":78.590},"bbox":{"west":78.588,"south":20.748,"east":78.592,"north":20.752},"geometry":{"type":"Polygon","coordinates":[[[78.588,20.748],[78.592,20.748],[78.592,20.752],[78.588,20.752],[78.588,20.748]]]},"area_m2":420.0,"zone_type":"residential","landuse":"residential","violation_type":"permitted_zone","model_confidence":0.55,"temporal_range":{"before_date":"2024-05-01","after_date":"2024-12-15"},"cadastral":{"within_parcel":True,"crosses_boundary":False,"parcel_id":"WRD-7783"},"status":"new"},
]

if not _alerts_loaded:
    @app.get("/api/alerts/geojson")
    async def fallback_alerts_geojson():
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": a["geometry"], "properties": {k: v for k, v in a.items() if k != "geometry"}}
            for a in DEMO_ALERTS
        ]}

    @app.get("/api/alerts/summary")
    async def fallback_alerts_summary():
        return {"total": len(DEMO_ALERTS), "avg_severity": 0.66, "max_severity": 0.92, "by_level": {"CRITICAL": 2, "HIGH": 2, "MEDIUM": 1, "LOW": 1}}

    @app.get("/api/alerts/{alert_id}")
    async def fallback_get_alert(alert_id: str):
        for a in DEMO_ALERTS:
            if a["alert_id"] == alert_id:
                return a
        return {"error": "not found"}

    @app.patch("/api/alerts/{alert_id}/status")
    async def fallback_update_status(alert_id: str, status: str = "reviewed"):
        for a in DEMO_ALERTS:
            if a["alert_id"] == alert_id:
                a["status"] = status
                return {"message": f"Updated to {status}"}
        return {"error": "not found"}

if not _analysis_loaded:
    @app.post("/api/analysis/")
    async def fallback_analysis():
        return {"analysis_id": "demo", "status": "completed", "message": "Demo mode — install rasterio/geopandas for real analysis", "num_alerts": len(DEMO_ALERTS)}

    @app.get("/api/analysis/{analysis_id}")
    async def fallback_analysis_status(analysis_id: str):
        return {"analysis_id": analysis_id, "status": "completed", "message": "Demo mode", "num_alerts": len(DEMO_ALERTS)}

# Serve dashboard static files
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")


# ── Root routes ──────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page."""
    index_path = dashboard_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "LandWatch AI API", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "gee_authenticated": False, "models_loaded": False, "demo_mode": not _alerts_loaded}


@app.get("/api/config")
async def get_config():
    return {
        "aoi": {"bbox": settings.aoi.bbox, "center": settings.aoi.center},
        "model": {"tile_size": settings.model.tile_size, "confidence_threshold": settings.model.confidence_threshold, "device": settings.model.device},
    }


@app.on_event("startup")
async def startup():
    settings.ensure_dirs()
    logger.info("LandWatch AI started")
    logger.info(f"AOI: {settings.aoi.bbox}")
    logger.info(f"Dashboard: http://localhost:{settings.api.port}")
    logger.info(f"API docs: http://localhost:{settings.api.port}/docs")
    if not _alerts_loaded:
        logger.info(f"DEMO MODE: {len(DEMO_ALERTS)} sample alerts loaded")


# ── Run directly ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent.parent.parent)],
    )
