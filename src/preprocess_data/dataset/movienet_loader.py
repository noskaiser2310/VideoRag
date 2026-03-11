"""
MovieNet Data Loader

Loads movie metadata, annotation scenes, and keyframe paths from
the MovieNet dataset structure.

Adapted from: scripts/movienet_loader.py + movienet_tools/metaio/
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class MovieNetLoader:
    """Load MovieNet annotation, metadata, and keyframe data."""

    def __init__(self):
        self.cache = {}

    #  Annotation Loading 

    def load_annotation(self, movie_id: str) -> Dict[str, Any]:
        """Load full annotation JSON for a movie (scenes, shots, frames)."""
        ann_path = Cfg.ANNOTATION_DIR / f"{movie_id}.json"
        if not ann_path.exists():
            return {}
        try:
            return json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse annotation for {movie_id}: {e}")
            return {}

    def load_scenes(self, movie_id: str, fps: float = 24.0) -> List[Dict]:
        """Load scenes with timestamp conversion."""
        data = self.load_annotation(movie_id)
        scenes = []
        for i, s in enumerate(data.get("scene", [])):
            frame_range = s.get("frame", [0, 0])
            shot_range = s.get("shot", [0, 0])
            if len(frame_range) < 2 or len(shot_range) < 2:
                continue

            start_sec = frame_range[0] / fps
            end_sec = frame_range[1] / fps

            scenes.append(
                {
                    "scene_idx": i,
                    "scene_id": s.get("id", f"scene_{i}"),
                    "shot_start": shot_range[0],
                    "shot_end": shot_range[1],
                    "frame_start": frame_range[0],
                    "frame_end": frame_range[1],
                    "start_seconds": round(start_sec, 2),
                    "end_seconds": round(end_sec, 2),
                    "duration_seconds": round(end_sec - start_sec, 2),
                    "place_tag": s.get("place_tag"),
                    "action_tag": s.get("action_tag"),
                }
            )
        return scenes

    #  Metadata Loading 

    def load_meta(self, movie_id: str) -> Dict:
        """Load movie metadata (title, genres, cast) from meta JSON."""
        for d in Cfg.META_SEARCH_DIRS:
            p = d / f"{movie_id}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    #  Keyframe Loading 

    def get_keyframe_paths(self, movie_id: str) -> List[str]:
        """Get all keyframe image paths for a movie."""
        for d in Cfg.KEYF_SEARCH_DIRS:
            movie_dir = d / movie_id
            if movie_dir.exists():
                return sorted(str(p) for p in movie_dir.glob("*.jpg"))
        return []

    def load_keyframe_index(self, movie_id: str) -> Optional[Dict]:
        """Load keyframe_index.json (from precision extraction)."""
        for d in Cfg.KEYF_SEARCH_DIRS:
            idx_path = d / movie_id / "keyframe_index.json"
            if idx_path.exists():
                try:
                    return json.loads(idx_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return None

    #  Discovery 

    def list_annotated_movies(self) -> List[str]:
        """List movies with annotation data."""
        if Cfg.ANNOTATION_DIR.exists():
            return sorted(p.stem for p in Cfg.ANNOTATION_DIR.glob("*.json"))
        return []

    def list_movies_with_keyframes(self) -> List[str]:
        """List movies that have keyframe directories."""
        ids = set()
        for d in Cfg.KEYF_SEARCH_DIRS:
            if d.exists():
                ids |= {
                    p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.jpg"))
                }
        return sorted(ids)
