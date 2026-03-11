import os
import json
import logging
from pathlib import Path
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(r"D:\Study\School\project_ky4\data")
SUBSET_DIR = Path(r"D:\Study\School\project_ky4\movie_data_subset_20")
SUBTITLE_DIR = SUBSET_DIR / "subtitle"


def parse_srt(content):
    """Simple SRT parser to extract text blocks."""
    blocks = re.split(r"\n\s*\n", content.strip())
    texts = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            # line 0: index, line 1: timestamp, line 2+: text
            text = " ".join(lines[2:]).strip()
            # Remove html tags like <i>
            text = re.sub(r"<[^>]+>", "", text)
            if text:
                texts.append(text)
    return texts


def build_graph_dataset(limit_per_movie=5):
    """
    Extracts subtitle chunks to feed into GraphIndexer.
    Limits chunks per movie to avoid massive API costs during testing.
    """
    if not SUBTITLE_DIR.exists():
        logger.error(f"Subtitle dir not found: {SUBTITLE_DIR}")
        return []

    documents = []
    srt_files = list(SUBTITLE_DIR.glob("*.srt"))
    logger.info(f"Found {len(srt_files)} subtitle files.")

    for srt_path in srt_files:
        imdb_id = srt_path.stem
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        texts = parse_srt(content)

        # Group every 20 lines into a "SceneChunk" to provide enough context for Graph RAG
        chunk_size = 20
        chunks = [
            " ".join(texts[i : i + chunk_size])
            for i in range(0, len(texts), chunk_size)
        ]

        # Apply a limit for testing/building speed
        chunks = chunks[:limit_per_movie]

        for idx, chunk in enumerate(chunks):
            documents.append(
                {
                    "text": chunk,
                    "movie_id": imdb_id,
                    "clip_id": f"{imdb_id}_chunk_{idx}",
                    "metadata": {"scene_label": f"Excerpt {idx}"},
                }
            )

    return documents


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(r"d:\Study\School\project_ky4\src")))

    from movierag.indexing.graph_indexer import GraphIndexer

    dataset = build_graph_dataset(limit_per_movie=5)  # ~80 chunks total

    indexer = GraphIndexer(
        index_dir=str(DATA_DIR / "unified_dataset"), index_name="movie_graph_index"
    )

    indexer.build_index(dataset)
    logger.info("Graph Index Build Complete.")
