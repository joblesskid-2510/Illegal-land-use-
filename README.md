<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Leaflet.js-1.9-199900?style=for-the-badge&logo=leaflet&logoColor=white" />
  <img src="https://img.shields.io/badge/PostGIS-16-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/GEE-Sentinel--2-4285F4?style=for-the-badge&logo=google-earth&logoColor=white" />
</p>

# 🛰️ LandWatch AI — Satellite-Driven Unauthorized Land Detection

> **A production-grade deep learning pipeline that analyzes multi-temporal satellite imagery to detect illegal land use, unauthorized construction, and regulatory violations using change detection, semantic segmentation, and GIS spatial analysis.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Models](#models)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Overview

Unauthorized land development — illegal construction on agricultural land, encroachment into protected forests, and violations of zoning regulations — is a growing challenge for urban planning authorities and environmental agencies worldwide.

**LandWatch AI** addresses this by building an end-to-end AI pipeline that:

1. **Ingests** multi-temporal satellite imagery (Sentinel-2, Landsat-8) via Google Earth Engine
2. **Detects changes** between time periods using a Siamese CNN architecture
3. **Classifies land cover** through DeepLabV3+ semantic segmentation (7 classes)
4. **Cross-references** detected changes against zoning regulations and cadastral boundaries
5. **Generates alerts** with severity scores, geographic coordinates, and visual evidence
6. **Visualizes results** on an interactive Leaflet.js dashboard with real-time filtering

---

## Key Features

| Feature | Description |
|---|---|
| 🔄 **Bi-Temporal Change Detection** | Siamese Network with shared ResNet-18 encoder identifies pixel-level changes between before/after satellite composites |
| 🗺️ **7-Class Land Cover Segmentation** | DeepLabV3+ (ResNet-50 backbone) segments: buildings, roads, vegetation, water, bare soil, construction, background |
| 📡 **Automated Satellite Ingestion** | Google Earth Engine integration with cloud masking, temporal compositing, and spectral index computation (NDVI, NDBI, NDWI, BSI) |
| ⚖️ **Regulatory Cross-Referencing** | Overlays detections against OSM land-use / zoning shapefiles to flag violations (agricultural, protected, water body zones) |
| 📐 **Cadastral Boundary Analysis** | Checks if detected changes cross parcel boundaries or fall outside registered parcels |
| 🚨 **Severity-Weighted Alerts** | Multi-factor scoring: zone sensitivity (40%), change confidence (25%), area (20%), boundary violations (15%) |
| 🖥️ **Interactive Dashboard** | Dark-themed glassmorphism UI with Leaflet.js map, real-time alert filtering, and analysis modal |
| 🐳 **Containerized Deployment** | Docker Compose with FastAPI app, PostGIS database, and MLflow experiment tracking |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LANDWATCH AI PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  INGESTION   │───▶│  INFERENCE   │───▶│   SPATIAL ANALYSIS   │   │
│  │              │    │              │    │                      │   │
│  │ • GEE Client │    │ • Siamese    │    │ • Raster → Vector    │   │
│  │ • Sentinel-2 │    │   CNN        │    │ • Zoning Overlay     │   │
│  │ • Landsat-8  │    │ • DeepLabV3+ │    │ • Cadastral Check    │   │
│  │ • Cloud Mask │    │ • Tile-based │    │ • Severity Scoring   │   │
│  │ • Composites │    │   Batching   │    │ • Alert Generation   │   │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘   │
│                                                     │               │
│  ┌──────────────────────────────────────────────────▼────────────┐  │
│  │                      SERVING LAYER                            │  │
│  │                                                               │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐  │  │
│  │  │  FastAPI     │   │  PostGIS     │   │  Leaflet.js       │  │  │
│  │  │  REST API    │   │  Database    │   │  Dashboard        │  │  │
│  │  │  /api/*      │   │  Alerts DB   │   │  Dark Theme       │  │  │
│  │  │  Tile Server │   │  Spatial Idx │   │  Real-time Alerts  │  │  │
│  │  └─────────────┘   └──────────────┘   └───────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Deep Learning** | PyTorch 2.x, torchvision, segmentation-models-pytorch |
| **Remote Sensing** | Google Earth Engine API, Rasterio, GDAL |
| **Geospatial** | GeoPandas, Shapely, PostGIS, OSMnx |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend** | Leaflet.js 1.9, Lucide Icons, Vanilla JS |
| **Database** | PostgreSQL 16 + PostGIS 3.4 |
| **MLOps** | MLflow 2.10, Docker, Docker Compose |
| **Config** | pydantic-settings, python-dotenv |

---

## Project Structure

```
landwatch-ai/
├── config/
│   └── settings.py              # Pydantic-based centralized configuration
├── src/
│   ├── ingestion/
│   │   ├── gee_client.py        # GEE authentication, collection fetch, cloud masking
│   │   ├── sentinel2.py         # Spectral indices (NDVI, NDBI), temporal pair selection
│   │   └── preprocessor.py      # Tiling, normalization, stitching, GeoTIFF I/O
│   ├── models/
│   │   ├── siamese_net.py       # Siamese CNN (ResNet-18) for bi-temporal change detection
│   │   ├── segmentation.py      # DeepLabV3+ (ResNet-50) for 7-class land cover
│   │   └── inference.py         # Unified batched tile inference + scene reconstruction
│   ├── gis/
│   │   ├── spatial_analysis.py  # Raster-to-vector, OSM fetch, zoning overlay
│   │   ├── cadastral.py         # Boundary crossing & parcel containment checks
│   │   └── alert_generator.py   # Severity scoring + GeoJSON serialization
│   ├── api/
│   │   ├── main.py              # FastAPI app, static files, fallback demo endpoints
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routes/
│   │       ├── analysis.py      # Background job orchestration
│   │       ├── alerts.py        # Alert CRUD + GeoJSON export
│   │       └── tiles.py         # Raster-to-PNG tile serving
│   └── dashboard/
│       ├── index.html           # Leaflet map + sidebar + modals
│       ├── style.css            # Dark glassmorphism theme
│       └── app.js               # Map interactions, alert rendering, API integration
├── Dockerfile                   # Python 3.11 + GDAL slim image
├── docker-compose.yml           # App + PostGIS + MLflow services
├── requirements.txt             # Pinned Python dependencies
├── .env                         # Environment configuration (not committed)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Google Earth Engine service account (JSON key)
- Docker & Docker Compose (for full deployment)
- GDAL system libraries (`brew install gdal` on macOS)

### Quick Start (Demo Mode)

Run the dashboard with demo alerts — no GEE or GDAL required:

```bash
# Clone
git clone https://github.com/joblesskid-2510/Illegal-land-use-.git
cd Illegal-land-use-

# Install minimal deps
pip install fastapi uvicorn pydantic pydantic-settings loguru python-dotenv python-multipart

# Start server
python -m src.api.main
```

Open **http://localhost:8000** — the dashboard loads with 6 demo alerts over Wardha, Maharashtra.

### Full Installation

```bash
# Install system GDAL
brew install gdal          # macOS
# sudo apt install gdal-bin libgdal-dev   # Ubuntu

# Install all Python deps
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GEE key path, PostGIS credentials, and AOI

# Start with Docker (PostGIS + MLflow + App)
docker-compose up -d
```

---

## Usage

### Running an Analysis

1. Open the dashboard at `http://localhost:8000`
2. Click **"New Analysis"** in the top-right
3. Set before/after time periods, cloud cover threshold, and confidence level
4. Click **"Run Analysis"** — the pipeline:
   - Fetches Sentinel-2 composites for both periods via GEE
   - Tiles and normalizes imagery
   - Runs Siamese CNN for change detection
   - Runs DeepLabV3+ for land cover segmentation
   - Cross-references changes against zoning regulations
   - Generates severity-scored alerts
5. Results appear as pulsing markers on the map with polygon overlays

### Filtering Alerts

- **Severity**: Toggle CRITICAL / HIGH / MEDIUM / LOW chips
- **Confidence**: Adjust the slider (0–100%)
- **Zone Type**: Filter by Agricultural, Protected, Residential, etc.
- Click any alert card or marker for detailed information

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/docs` | Swagger API documentation |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/config` | Current AOI and model configuration |
| `POST` | `/api/analysis/` | Trigger new satellite analysis |
| `GET` | `/api/analysis/{id}` | Poll analysis status |
| `GET` | `/api/alerts/geojson` | All alerts as GeoJSON FeatureCollection |
| `GET` | `/api/alerts/summary` | Aggregate alert statistics |
| `GET` | `/api/alerts/{id}` | Single alert detail |
| `PATCH` | `/api/alerts/{id}/status` | Update alert status (reviewed/false_positive) |
| `GET` | `/api/tiles/{layer}/{z}/{x}/{y}.png` | Raster tile serving |

---

## Dashboard

The interactive dashboard features:

- 🗺️ **Satellite basemap** (Esri World Imagery) with label overlay
- 📍 **Pulsing alert markers** color-coded by severity (red/orange/yellow/green)
- 🔲 **Polygon overlays** showing exact detected change boundaries
- 📊 **Real-time stats** (total alerts, critical count, affected area)
- 🎚️ **Filter controls** for severity, confidence, and zone type
- 📋 **Alert detail panel** with severity bar, coordinates, cadastral info, and action buttons
- 🌙 **Dark glassmorphism UI** with smooth animations

---

## Models

### Siamese Change Detector

| Property | Value |
|---|---|
| Architecture | Siamese Network with shared encoder |
| Backbone | ResNet-18 (pretrained on ImageNet) |
| Input | Two 256×256 patches (before/after) |
| Output | Binary change mask |
| Method | Feature differencing + decoder head |

### Land Cover Segmenter

| Property | Value |
|---|---|
| Architecture | DeepLabV3+ |
| Backbone | ResNet-50 (pretrained on ImageNet) |
| Input | 256×256 satellite patch |
| Output | 7-class segmentation mask |
| Classes | Background, Building, Road, Vegetation, Water, Bare Soil, Construction |

### Severity Scoring

```
Score = 0.40 × zone_weight + 0.25 × model_confidence + 0.20 × area_weight + 0.15 × boundary_penalty

Zone weights: water/protected → 1.0, agricultural → 0.8, green_space → 0.6, residential → 0.3
```

---

## Future Roadmap

- [ ] Fine-tune models on high-resolution regional satellite data
- [ ] Add ChangeFormer (transformer-based) as alternative change detector
- [ ] Implement user-drawn AOI selection on the map
- [ ] Add temporal animation showing change progression
- [ ] PDF report generation with before/after comparisons
- [ ] Role-based access control for multi-agency deployment
- [ ] Integration with India's Bhuvan portal for cadastral data
- [ ] Mobile-responsive dashboard for field inspectors

---

## License

This project is developed for academic and research purposes.

---

<p align="center">
  <strong>Built with 🛰️ by LandWatch AI Team — 2025</strong>
</p>
