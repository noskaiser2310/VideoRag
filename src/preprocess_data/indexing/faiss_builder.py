"""
FAISS Index Builder

Reads temporal chunks (all_chunks.json) and builds a FAISS visual index
with enriched metadata for each keyframe.

Adapted from: scripts/reindex_temporal.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class FaissBuilder:
    """Build FAISS visual index from temporal chunks."""

    def __init__(self, index_dir: str = None):
        self.index_dir = Path(index_dir) if index_dir else Cfg.INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def build(self, chunks_path: Path = None) -> Dict:
        """
        Read all_chunks.json, flatten to per-keyframe items,
        encode with CLIP, and build FAISS index.
        """
        chunks_path = chunks_path or (Cfg.TEMPORAL_CHUNKS_DIR / "all_chunks.json")

        # Load chunks
        logger.info(f"Loading temporal chunks from: {chunks_path}")
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = data.get("chunks", [])
        logger.info(f"  Total chunks: {len(chunks)}")

        # Flatten: one item per keyframe
        items = []
        for chunk in chunks:
            for kf_path in chunk.get("keyframe_paths", []):
                items.append(
                    {
                        "id": f"{chunk['chunk_id']}_{Path(kf_path).stem}",
                        "path": kf_path,
                        "movie_id": chunk["movie_id"],
                        "chunk_id": chunk["chunk_id"],
                        "start_time": chunk.get("start_time", ""),
                        "end_time": chunk.get("end_time", ""),
                        "start_seconds": chunk.get("start_seconds", 0),
                        "end_seconds": chunk.get("end_seconds", 0),
                        "description": chunk.get("description", ""),
                        "dialogue_text": chunk.get("dialogue_text", ""),
                        "characters": chunk.get("characters", []),
                        "cast_in_scene": chunk.get("cast_in_scene", []),
                        "situation": chunk.get("situation", ""),
                        "scene_label": chunk.get("scene_label", ""),
                        "timestamp_source": chunk.get("timestamp_source", ""),
                        "title": chunk.get("title", ""),
                    }
                )

        logger.info(f"  Total keyframe items: {len(items)}")
        if not items:
            logger.warning("  No items to index!")
            return {"items": 0}

        # Import indexer
        sys.path.insert(0, str(Cfg.SRC_DIR))
        from movierag.indexing.visual_indexer import VisualIndexer

        indexer = VisualIndexer(str(self.index_dir))
        indexer.build_index(items, id_key="id", path_key="path")
        indexer.save()

        logger.info(f"   FAISS index built: {len(items)} vectors → {self.index_dir}")
        return {"items": len(items), "index_dir": str(self.index_dir)}

    def build_incremental(self, movie_id: str) -> Dict:
        """
        Add a single movie's chunks to the existing FAISS index.
        For new video ingest without rebuilding the entire index.
        """
        chunks_path = Cfg.TEMPORAL_CHUNKS_DIR / f"{movie_id}_chunks.json"
        if not chunks_path.exists():
            logger.warning(f"No chunks found for {movie_id}")
            return {"items": 0}

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        items = []
        for chunk in chunks:
            for kf_path in chunk.get("keyframe_paths", []):
                items.append(
                    {
                        "id": f"{chunk['chunk_id']}_{Path(kf_path).stem}",
                        "path": kf_path,
                        "movie_id": chunk["movie_id"],
                        "chunk_id": chunk["chunk_id"],
                        "start_time": chunk.get("start_time", ""),
                        "end_time": chunk.get("end_time", ""),
                        "start_seconds": chunk.get("start_seconds", 0),
                        "end_seconds": chunk.get("end_seconds", 0),
                        "description": chunk.get("description", ""),
                        "dialogue_text": chunk.get("dialogue_text", ""),
                        "characters": chunk.get("characters", []),
                        "cast_in_scene": chunk.get("cast_in_scene", []),
                        "situation": chunk.get("situation", ""),
                        "timestamp_source": chunk.get("timestamp_source", ""),
                        "title": chunk.get("title", ""),
                    }
                )

        if not items:
            return {"items": 0}

        sys.path.insert(0, str(Cfg.SRC_DIR))
        from movierag.indexing.visual_indexer import VisualIndexer

        indexer = VisualIndexer(str(self.index_dir))
        # Load existing index, then add new items
        if indexer.load():
            logger.info(
                f"  Loaded existing index, adding {len(items)} items for {movie_id}"
            )

        indexer.build_index(items, id_key="id", path_key="path")
        indexer.save()

        logger.info(f"   Added {len(items)} vectors for {movie_id}")
        return {"items": len(items), "movie_id": movie_id}
