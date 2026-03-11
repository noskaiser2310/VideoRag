"""
Unified Dataset Builder

Merges MovieNet keyframes + MovieGraphs scene descriptions into
a single JSON dataset file.

Adapted from: scripts/build_unified_dataset.py
"""

import json
import logging
from typing import List, Dict

from ..config import PreprocessConfig as Cfg
from .movienet_loader import MovieNetLoader
from .moviegraph_loader import MovieGraphLoader

logger = logging.getLogger(__name__)


class UnifiedDatasetBuilder:
    """Build the unified movierag_dataset.json from MovieNet + MovieGraphs."""

    def __init__(self):
        self.movienet = MovieNetLoader()
        self.moviegraphs = MovieGraphLoader()

    def build(self, movie_ids: List[str] = None) -> Dict:
        """
        Build unified dataset for given movies (or discover all).
        Returns the dataset dict and saves to disk.
        """
        if movie_ids is None:
            movie_ids = Cfg.get_all_movie_ids()

        logger.info(f"Building unified dataset for {len(movie_ids)} movies...")

        dataset = {
            "metadata": {
                "name": "MovieRAG Unified Dataset",
                "version": "2.0",
                "description": "Merged MovieNet (keyframes) + MovieGraphs (scene descriptions)",
                "num_movies": 0,
                "total_clips": 0,
                "total_keyframes": 0,
            },
            "movies": {},
        }

        total_clips = 0
        total_keyframes = 0

        for mid in movie_ids:
            logger.info(f"  Processing {mid}...")
            meta = self.movienet.load_meta(mid)

            movie_entry = {
                "imdb_id": mid,
                "title": meta.get("title", mid),
                "year": meta.get("year"),
                "sources": {"movienet": False, "moviegraphs": False},
                "keyframes": [],
                "clips": [],
            }

            # MovieNet keyframes
            keyframes = self.movienet.get_keyframe_paths(mid)
            if keyframes:
                movie_entry["keyframes"] = [
                    {"path": p, "filename": p.split("\\")[-1].split("/")[-1]}
                    for p in keyframes
                ]
                movie_entry["sources"]["movienet"] = True
                total_keyframes += len(keyframes)

            # MovieGraphs clips
            clips = self.moviegraphs.extract_clips(mid)
            if clips:
                movie_entry["clips"] = clips
                movie_entry["sources"]["moviegraphs"] = True
                total_clips += len(clips)

            # Cast
            cast = self.moviegraphs.extract_cast(mid)
            if cast:
                movie_entry["cast"] = cast

            dataset["movies"][mid] = movie_entry

        dataset["metadata"]["num_movies"] = len(dataset["movies"])
        dataset["metadata"]["total_clips"] = total_clips
        dataset["metadata"]["total_keyframes"] = total_keyframes

        # Save
        Cfg.UNIFIED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        output_path = Cfg.UNIFIED_DATASET_JSON
        output_path.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"   Saved unified dataset: {output_path}")
        logger.info(
            f"     Movies: {dataset['metadata']['num_movies']}, "
            f"Clips: {total_clips}, Keyframes: {total_keyframes}"
        )

        return dataset
