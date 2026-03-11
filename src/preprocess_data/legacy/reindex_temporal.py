"""
 Re-Index FAISS with Temporal Chunks

Reads the temporal chunks from `data/temporal_chunks/all_chunks.json` and
rebuilds the FAISS visual index where each keyframe entry carries the full
5-layer metadata (exact timestamps, description, dialogue, characters, cast).

This replaces the old index that only had keyframe paths + basic shot IDs.

Usage:
    python scripts/reindex_temporal.py
    python scripts/reindex_temporal.py --index-dir data/indexes_v2
    python scripts/reindex_temporal.py --movie tt0120338

Output:
    data/indexes/visual_index.faiss         ← FAISS binary index
    data/indexes/visual_index_metadata.json ← Enriched metadata per frame
"""

import json
import sys
import logging
import argparse
from pathlib import Path

# Ensure imports work
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from movierag.indexing.visual_indexer import VisualIndexer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ReindexTemporal")

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = BASE_DIR / "temporal_chunks" / "all_chunks.json"
DEFAULT_INDEX_DIR = BASE_DIR / "indexes"


def flatten_chunks_to_items(chunks: list, movie_filter: str = None) -> list:
    """
    Flatten temporal chunks into per-keyframe items for FAISS indexing.

    Each chunk may contain multiple keyframes. We create one FAISS entry
    per keyframe, but attach the FULL chunk metadata so the pipeline can
    retrieve exact timestamps and descriptions directly from the search result.
    """
    items = []
    skipped_no_keyframes = 0

    for chunk in chunks:
        movie_id = chunk["movie_id"]

        if movie_filter and movie_id != movie_filter:
            continue

        keyframe_paths = chunk.get("keyframe_paths", [])

        if not keyframe_paths:
            skipped_no_keyframes += 1
            continue

        # Build the shared context that every keyframe in this chunk inherits
        chunk_context = {
            "chunk_id": chunk["chunk_id"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "start_seconds": chunk["start_seconds"],
            "end_seconds": chunk["end_seconds"],
            "duration_seconds": chunk["duration_seconds"],
            "timestamp_source": chunk["timestamp_source"],
            "description": chunk.get("description", ""),
            "situation": chunk.get("situation", ""),
            "scene_label": chunk.get("scene_label", ""),
            "characters": chunk.get("characters", []),
            "dialogue_text": chunk.get("dialogue_text", ""),
            "title": chunk.get("title", ""),
            "genres": chunk.get("genres", []),
            "cast_in_scene": chunk.get("cast_in_scene", []),
            "shot_start": chunk.get("shot_start", 0),
            "shot_end": chunk.get("shot_end", 0),
        }

        for kf_path in keyframe_paths:
            # Validate path exists
            if not Path(kf_path).exists():
                continue

            # Extract a unique ID from the path: e.g. tt0120338_shot_0009_img_1
            fname = Path(kf_path).stem  # shot_0009_img_1
            kf_id = f"{movie_id}_{fname}"

            item = {
                "keyframe_id": kf_id,
                "keyframe_path": kf_path,
                "movie_id": movie_id,
                **chunk_context,
            }
            items.append(item)

    if skipped_no_keyframes > 0:
        logger.info(f"  Skipped {skipped_no_keyframes} chunks with no keyframes")

    return items


def main():
    parser = argparse.ArgumentParser(description="Re-index FAISS with Temporal Chunks")
    parser.add_argument(
        "--chunks-path",
        type=str,
        default=str(CHUNKS_PATH),
        help="Path to all_chunks.json",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(DEFAULT_INDEX_DIR),
        help="Directory to save the new FAISS index",
    )
    parser.add_argument(
        "--movie",
        type=str,
        default=None,
        help="Optional: only re-index a single movie (e.g. tt0120338)",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="visual_index",
        help="Base name for the index files (default: visual_index)",
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks_path)
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.error("Run build_temporal_chunks.py first to generate temporal chunks.")
        return

    # Load chunks
    logger.info(f"Loading temporal chunks from: {chunks_path}")
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    meta = data.get("metadata", {})

    logger.info(f"  Total chunks:   {meta.get('total_chunks', len(chunks))}")
    logger.info(f"  Total movies:   {meta.get('total_movies', '?')}")
    logger.info(f"  With timestamps: {meta.get('chunks_with_exact_timestamps', '?')}")
    logger.info(f"  With dialogue:   {meta.get('chunks_with_dialogue', '?')}")

    # Flatten to per-keyframe items
    logger.info("\nFlattening chunks to per-keyframe items...")
    items = flatten_chunks_to_items(chunks, movie_filter=args.movie)

    if not items:
        logger.error("No valid items to index! Check if keyframe paths exist.")
        return

    # Count unique movies
    movie_ids = set(item["movie_id"] for item in items)
    logger.info(f"  Total items:  {len(items)} keyframes")
    logger.info(f"  Movies:       {len(movie_ids)}")
    logger.info(f"  Sample item:  {items[0]['keyframe_id']}")

    # Build enriched FAISS index
    logger.info(f"\nBuilding FAISS index at: {args.index_dir}")
    indexer = VisualIndexer(index_dir=args.index_dir, index_name=args.index_name)
    indexer.build_index(items)

    # Print stats
    stats = indexer.get_statistics()
    logger.info(f"\nIndex statistics: {stats}")

    # Verify metadata richness
    test_meta = indexer._metadata[0] if indexer._metadata else {}
    has_temporal = "start_time" in test_meta and "end_time" in test_meta
    has_description = bool(test_meta.get("description"))
    has_dialogue = bool(test_meta.get("dialogue_text"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"   RE-INDEX COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total vectors:            {stats.get('total_vectors', '?')}")
    logger.info(f"  Movies indexed:           {len(movie_ids)}")
    logger.info(f"  Metadata has timestamps:  {'' if has_temporal else ''}")
    logger.info(f"  Metadata has description: {'' if has_description else ''}")
    logger.info(f"  Metadata has dialogue:    {'' if has_dialogue else ''}")
    logger.info(f"\n  Index files:")
    logger.info(f"    {args.index_dir}/{args.index_name}.faiss")
    logger.info(f"    {args.index_dir}/{args.index_name}_metadata.json")


if __name__ == "__main__":
    main()
