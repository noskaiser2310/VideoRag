"""
Unified Dataset Builder for Flow 1.
Parses multi-modal data from annotation, meta, script, and subtitle directories.
Outputs a unified JSON dataset to data/rawdata/unified_dataset.json.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from tqdm import tqdm

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from movierag.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_srt(srt_path: Path):
    """Parse an SRT file into a list of subtitle dictionaries."""
    subtitles = []
    if not srt_path.exists():
        return subtitles

    try:
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Basic SRT parsing
        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:
                # index = lines[0]
                timestamps = lines[1]
                text = " ".join(lines[2:])
                # Clean HTML tags and excessive whitespace
                text = re.sub(r"<[^>]+>", "", text).strip()
                if text:
                    subtitles.append({"timestamps": timestamps, "text": text})
    except Exception as e:
        logger.warning(f"Error parsing {srt_path}: {e}")

    return subtitles


def parse_script(script_path: Path):
    """Parse a .script file and extract dialogue."""
    script_text = ""
    if not script_path.exists():
        return script_text

    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            script_text = f.read()
    except Exception as e:
        logger.warning(f"Error parsing {script_path}: {e}")

    return script_text


def build_dataset():
    config = Config.from_env()

    # Ensure rawdata dir exists
    config.paths.rawdata_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.paths.rawdata_dir / "unified_dataset.json"

    meta_dir = config.paths.meta_dir
    annotation_dir = config.paths.annotation_dir
    script_dir = config.paths.script_dir
    subtitle_dir = config.paths.subtitle_dir

    # Get all movie IDs based on meta directory
    if not meta_dir.exists():
        logger.error(f"Meta directory not found at {meta_dir}")
        return

    movie_files = list(meta_dir.glob("*.json"))
    logger.info(f"Found {len(movie_files)} movies in meta directory.")

    dataset = {
        "metadata": {
            "name": "MovieRAG Flow 1 Dataset",
            "version": "1.0",
            "description": "Merged dataset from annotations, meta, scripts, and subtitles",
            "num_movies": 0,
        },
        "movies": {},
    }

    for meta_file in tqdm(movie_files, desc="Processing Movies"):
        imdb_id = meta_file.stem

        # Load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Initialize movie entry
        movie_entry = {
            "imdb_id": imdb_id,
            "title": meta_data.get("title", ""),
            "genres": meta_data.get("genres", []),
            "cast": meta_data.get("cast", []),
            "plot": meta_data.get("plot", ""),
            "overview": meta_data.get("overview", ""),
            "storyline": meta_data.get("storyline", ""),
            "sources": {
                "meta": True,
                "annotation": False,
                "script": False,
                "subtitle": False,
                "moviegraph": False,
            },
        }

        # Load annotations
        anno_file = annotation_dir / f"{imdb_id}.json"
        if anno_file.exists():
            try:
                with open(anno_file, "r", encoding="utf-8") as f:
                    anno_data = json.load(f)

                # Store annotations
                # Only keep important fields to save space
                movie_entry["annotations"] = anno_data.get("cast", [])
                movie_entry["sources"]["annotation"] = True
            except Exception as e:
                logger.warning(f"Error reading annotation for {imdb_id}: {e}")

        # Load scripts
        script_file = script_dir / f"{imdb_id}.script"
        if script_file.exists():
            script_text = parse_script(script_file)
            if script_text:
                # We store a snippet or full text depending on requirements (storing full can be large)
                # First 2000 chars as summary, full in another structure if needed.
                movie_entry["script"] = script_text
                movie_entry["sources"]["script"] = True

        # Load subtitles
        srt_file = subtitle_dir / f"{imdb_id}.srt"
        if srt_file.exists():
            subtitles = parse_srt(srt_file)
            if subtitles:
                movie_entry["subtitles"] = subtitles
                movie_entry["sources"]["subtitle"] = True

        # Load MovieGraph
        graph_dir = config.paths.rawdata_dir / "moviegraph"
        graph_file = graph_dir / f"{imdb_id}.json"

        # Also check scene-level moviegraphs
        if graph_file.exists():
            try:
                with open(graph_file, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                movie_entry["moviegraph"] = graph_data
                movie_entry["sources"]["moviegraph"] = True
            except Exception as e:
                logger.warning(f"Error reading moviegraph for {imdb_id}: {e}")

        dataset["movies"][imdb_id] = movie_entry

    dataset["metadata"]["num_movies"] = len(dataset["movies"])

    logger.info(f"Saving unified dataset to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f" Unified Dataset Created Successfully!")
    logger.info(f"Movies Processed: {dataset['metadata']['num_movies']}")
    logger.info(f"File Size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    build_dataset()
