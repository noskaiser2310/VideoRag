"""
MovieGraphs Data Loader

Loads MovieGraphs scene graph data from the all_movies.pkl pickle file.
Provides structured access to clip descriptions, characters, and scene graphs.

Adapted from: data/MovieGraphs_repo/py3loader_new/GraphClasses.py
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class MovieGraphLoader:
    """Load and query MovieGraphs scene graph data."""

    def __init__(self):
        self._data = None

    def _ensure_loaded(self):
        """Lazy-load the pickle data."""
        if self._data is not None:
            return

        pkl_path = Cfg.MOVIEGRAPHS_PKL
        if not pkl_path.exists():
            logger.warning(f"MovieGraphs pkl not found: {pkl_path}")
            self._data = {}
            return

        # Add the loader dir to sys.path for unpickling GraphClasses
        loader_dir = str(Cfg.MOVIEGRAPHS_DIR)
        if loader_dir not in sys.path:
            sys.path.insert(0, loader_dir)

        logger.info(f"Loading MovieGraphs from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self._data = pickle.load(f)
        logger.info(f"  Loaded {len(self._data)} movies from MovieGraphs")

    def list_movies(self) -> List[str]:
        """List all available movie IDs."""
        self._ensure_loaded()
        return sorted(self._data.keys())

    def get_movie(self, movie_id: str) -> Optional[Any]:
        """Get raw MovieGraph object for a movie."""
        self._ensure_loaded()
        return self._data.get(movie_id)

    def extract_clips(self, movie_id: str) -> List[Dict]:
        """Extract structured clip info from MovieGraphs data."""
        movie = self.get_movie(movie_id)
        if movie is None:
            return []

        clip_graphs = getattr(movie, "clip_graphs", {})
        if not clip_graphs:
            for attr in ["clipgraphs", "clips", "scenes"]:
                clip_graphs = getattr(movie, attr, {})
                if clip_graphs:
                    break

        clips = []
        for sid, cg in clip_graphs.items():
            clip_info = self._extract_clip_info(cg)
            clip_info["clip_id"] = str(sid)
            clips.append(clip_info)

        return clips

    def extract_cast(self, movie_id: str) -> List[Dict]:
        """Extract cast list for a movie."""
        movie = self.get_movie(movie_id)
        if movie is None:
            return []

        castlist = getattr(movie, "castlist", None)
        if castlist and isinstance(castlist, list):
            return [
                {"name": c.get("name", ""), "id": c.get("chid", "")} for c in castlist
            ]
        return []

    @staticmethod
    def _extract_clip_info(clip_graph) -> Dict:
        """Extract structured info from a ClipGraph object."""
        info = {
            "situation": getattr(clip_graph, "situation", ""),
            "scene_label": getattr(clip_graph, "scene_label", ""),
            "description": getattr(clip_graph, "description", ""),
        }

        # Video/shot info
        video = getattr(clip_graph, "video", {})
        if video:
            info["start_shot"] = video.get("ss", None)
            info["end_shot"] = video.get("es", None)
            info["scene_ids"] = video.get("scene", [])

        # Characters
        try:
            chars = clip_graph.get_characters()
            info["characters"] = (
                [{"name": c[0], "id": c[1]} for c in chars] if chars else []
            )
        except Exception:
            info["characters"] = []

        # Entities, attributes, interactions
        try:
            node_dict = clip_graph.get_node_type_dict()
            info["entities"] = node_dict.get("entity", [])
            info["attributes"] = list(set(node_dict.get("attribute", [])))
            info["interactions"] = node_dict.get("interaction", [])
            info["relationships"] = node_dict.get("relationship", [])
        except Exception:
            pass

        return info
