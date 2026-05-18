"""
Analysis routes — trigger satellite analysis and retrieve results.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from config.settings import settings
from src.api.schemas import AnalysisRequest, AnalysisResponse, AlertSummary
from src.ingestion.gee_client import gee_client
from src.models.inference import pipeline
from src.gis.spatial_analysis import spatial_analyzer
from src.gis.cadastral import cadastral_checker
from src.gis.alert_generator import alert_generator

router = APIRouter(prefix="/analysis", tags=["analysis"])

# In-memory job tracker
_jobs: dict[str, dict] = {}


def _run_analysis(job_id: str, request: AnalysisRequest):
    """Background task that runs the full analysis pipeline."""
    try:
        _jobs[job_id]["status"] = "running"
        logger.info(f"Starting analysis {job_id}")

        bbox = (
            [request.bbox.west, request.bbox.south, request.bbox.east, request.bbox.north]
            if request.bbox else settings.aoi.bbox
        )

        # 1. Download satellite imagery
        logger.info("Step 1: Downloading satellite imagery...")
        t1_composite = gee_client.get_composite(
            bbox, request.t1_start, request.t1_end,
            max_cloud_pct=request.max_cloud_pct,
        )
        t2_composite = gee_client.get_composite(
            bbox, request.t2_start, request.t2_end,
            max_cloud_pct=request.max_cloud_pct,
        )

        output_dir = settings.raw_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        t1_path = output_dir / "t1_composite.tif"
        t2_path = output_dir / "t2_composite.tif"

        gee_client.export_to_local(
            t1_composite, t1_path, bbox,
            bands=["B2", "B3", "B4", "B8", "B11", "B12"],
        )
        gee_client.export_to_local(
            t2_composite, t2_path, bbox,
            bands=["B2", "B3", "B4", "B8", "B11", "B12"],
        )

        # 2. Run inference
        logger.info("Step 2: Running model inference...")
        results_dir = settings.processed_dir / job_id
        analysis_results = pipeline.full_analysis(t1_path, t2_path, results_dir)

        # 3. Spatial analysis
        logger.info("Step 3: Spatial analysis...")
        change_polygons = spatial_analyzer.raster_to_polygons(
            results_dir / "change_mask.tif",
            threshold=request.confidence_threshold,
        )

        if len(change_polygons) == 0:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["message"] = "No significant changes detected"
            _jobs[job_id]["num_alerts"] = 0
            return

        # 4. Zoning overlay
        overlaid = spatial_analyzer.overlay_with_zoning(change_polygons)
        flagged = spatial_analyzer.identify_violations(overlaid)

        # 5. Cadastral check
        flagged = cadastral_checker.check_boundary_violations(flagged)

        # 6. Generate alerts
        logger.info("Step 4: Generating alerts...")
        alerts = alert_generator.generate_alerts(
            flagged,
            t1_date=request.t1_start,
            t2_date=request.t2_end,
            analysis_id=job_id,
        )

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["message"] = f"Analysis complete. {len(alerts)} alerts generated."
        _jobs[job_id]["num_alerts"] = len(alerts)
        _jobs[job_id]["summary"] = alert_generator.get_summary()

        logger.info(f"Analysis {job_id} completed: {len(alerts)} alerts")

    except Exception as e:
        logger.error(f"Analysis {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["message"] = str(e)


@router.post("/", response_model=AnalysisResponse)
async def trigger_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a new satellite analysis job."""
    job_id = str(uuid.uuid4())[:8]

    _jobs[job_id] = {
        "status": "queued",
        "message": "Analysis queued",
        "num_alerts": 0,
        "summary": None,
    }

    background_tasks.add_task(_run_analysis, job_id, request)

    return AnalysisResponse(
        analysis_id=job_id,
        status="queued",
        message="Analysis has been queued and will run in the background",
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis job."""
    if analysis_id not in _jobs:
        raise HTTPException(status_code=404, detail="Analysis not found")

    job = _jobs[analysis_id]
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=job["status"],
        message=job["message"],
        num_alerts=job.get("num_alerts", 0),
        summary=AlertSummary(**job["summary"]) if job.get("summary") else None,
    )


@router.get("/")
async def list_analyses():
    """List all analysis jobs."""
    return {
        job_id: {
            "status": job["status"],
            "num_alerts": job.get("num_alerts", 0),
        }
        for job_id, job in _jobs.items()
    }
