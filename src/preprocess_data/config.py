"""
Shared configuration and paths for preprocess_data pipeline.
"""

import os
from pathlib import Path


class PreprocessConfig:
    """Central configuration for all preprocessing paths and settings."""

    #  Root directories 
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project_ky4/
    DATA_DIR = PROJECT_ROOT / "data"
    SRC_DIR = PROJECT_ROOT / "src"

    #  Raw inputs 
    RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
    RAW_MOVIES_DIR = DATA_DIR / "Raw_Movies"

    #  MovieNet data 
    MOVIENET_DIR = DATA_DIR / "movienet"
    MOVIENET_SUBSET_DIR = DATA_DIR / "movienet_subset"
    ANNOTATION_DIR = MOVIENET_SUBSET_DIR / "annotation"
    SUBTITLE_DIR = MOVIENET_SUBSET_DIR / "subtitle"
    META_DIR = MOVIENET_SUBSET_DIR / "meta"
    SHOT_KEYF_DIR = MOVIENET_DIR / "shot_keyf"
    STANDALONE_KEYF_DIR = DATA_DIR / "Standalone_Dataset" / "shot_keyf"

    #  Unified dataset 
    UNIFIED_DATASET_DIR = DATA_DIR / "unified_dataset"
    UNIFIED_DATASET_JSON = UNIFIED_DATASET_DIR / "movierag_dataset.json"

    #  MovieGraphs 
    MOVIEGRAPHS_DIR = DATA_DIR / "MovieGraphs_repo" / "py3loader_new"
    MOVIEGRAPHS_PKL = MOVIEGRAPHS_DIR / "all_movies.pkl"

    #  Temporal chunks output 
    TEMPORAL_CHUNKS_DIR = DATA_DIR / "temporal_chunks"

    #  Index output 
    INDEX_DIR = DATA_DIR / "indexes"
    GRAPH_PATH = UNIFIED_DATASET_DIR / "movie_graph_index.graphml"

    #  Video processing settings 
    KEYFRAME_HEIGHT = 720  # Target height for extracted keyframes
    KEYFRAME_QUALITY = 2  # JPEG quality (1-31, lower = better)
    KEYFRAME_INTERVAL_SEC = 3.0  # Fallback interval for old-style extraction

    #  Meta directories to search (in priority order) 
    META_SEARCH_DIRS = [
        MOVIENET_SUBSET_DIR / "meta",
        UNIFIED_DATASET_DIR / "meta",
    ]

    #  Keyframe search directories (in priority order) 
    KEYF_SEARCH_DIRS = [
        MOVIENET_DIR / "shot_keyf",
        DATA_DIR / "Standalone_Dataset" / "shot_keyf",
    ]

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        for d in [
            cls.TEMPORAL_CHUNKS_DIR,
            cls.INDEX_DIR,
            cls.SHOT_KEYF_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_video_path(cls, movie_id: str) -> Path | None:
        """Find the raw video file for a movie."""
        for d in [cls.RAW_VIDEOS_DIR, cls.RAW_MOVIES_DIR]:
            for ext in [".mp4", ".mkv", ".avi", ".mov"]:
                p = d / f"{movie_id}{ext}"
                if p.exists():
                    return p
        return None

    @classmethod
    def get_all_movie_ids(cls) -> list[str]:
        """Discover all movie IDs from annotation + unified dataset + videos."""
        ids = set()
        if cls.ANNOTATION_DIR.exists():
            ids |= {p.stem for p in cls.ANNOTATION_DIR.glob("*.json")}
        if cls.UNIFIED_DATASET_JSON.exists():
            import json

            data = json.loads(cls.UNIFIED_DATASET_JSON.read_text(encoding="utf-8"))
            ids |= set(data.get("movies", {}).keys())
        if cls.RAW_VIDEOS_DIR.exists():
            ids |= {
                p.stem
                for p in cls.RAW_VIDEOS_DIR.glob("*.*")
                if p.suffix in {".mp4", ".mkv", ".avi", ".mov"}
            }
        return sorted(ids)
