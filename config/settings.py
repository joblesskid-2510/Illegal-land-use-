"""
Centralized configuration using Pydantic Settings.
Reads from .env file and environment variables.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GEESettings(BaseSettings):
    """Google Earth Engine configuration."""
    service_account_key: str = Field(
        default="./illegal-land-75241315c879.json",
        alias="GEE_SERVICE_ACCOUNT_KEY",
    )
    project_id: str = Field(default="illegal-land", alias="GEE_PROJECT_ID")

    @property
    def key_path(self) -> Path:
        p = Path(self.service_account_key)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


class DatabaseSettings(BaseSettings):
    """PostGIS database configuration."""
    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    db: str = Field(default="landwatch", alias="POSTGRES_DB")
    user: str = Field(default="landwatch", alias="POSTGRES_USER")
    password: str = Field(default="landwatch_secret_2024", alias="POSTGRES_PASSWORD")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class AOISettings(BaseSettings):
    """Area of Interest bounding box."""
    west: float = Field(default=77.45, alias="AOI_WEST")
    south: float = Field(default=12.85, alias="AOI_SOUTH")
    east: float = Field(default=77.75, alias="AOI_EAST")
    north: float = Field(default=13.05, alias="AOI_NORTH")

    @property
    def bbox(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)


class APISettings(BaseSettings):
    """FastAPI server configuration."""
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")


class ModelSettings(BaseSettings):
    """ML model configuration."""
    # Siamese Network
    siamese_encoder: str = "resnet18"
    siamese_weights_path: Optional[str] = None
    # Segmentation
    seg_encoder: str = "resnet50"
    seg_classes: int = 6
    seg_weights_path: Optional[str] = None
    # Inference
    tile_size: int = 256
    tile_overlap: int = 32
    batch_size: int = 8
    device: str = "cpu"  # "cuda" if GPU available
    confidence_threshold: float = 0.5


class Settings(BaseSettings):
    """Root settings aggregator."""
    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}

    gee: GEESettings = GEESettings()
    database: DatabaseSettings = DatabaseSettings()
    aoi: AOISettings = AOISettings()
    api: APISettings = APISettings()
    model: ModelSettings = ModelSettings()

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    shapefile_dir: Path = PROJECT_ROOT / "data" / "shapefiles"
    sample_dir: Path = PROJECT_ROOT / "data" / "samples"
    model_dir: Path = PROJECT_ROOT / "models"

    def ensure_dirs(self):
        """Create all required directories."""
        for d in [self.raw_dir, self.processed_dir, self.shapefile_dir,
                  self.sample_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton
settings = Settings()
