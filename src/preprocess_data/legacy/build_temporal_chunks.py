"""
 Temporal Chunk Builder — 5-Layer Super Chunk Extraction

Merges 5 data sources into a unified temporal chunk index:
  ① annotation/*.json  → Exact frame boundaries (frame → timestamp via FPS)
  ② movierag_dataset.json → Clip semantics (description, characters, situation)
  ③ subtitle/*.srt → Dialogue aligned to time ranges
  ④ meta/*.json → Movie metadata (cast, genres, title)
  ⑤ shot_keyf/*/*.jpg → Representative keyframe paths for CLIP embedding

Output: data/temporal_chunks/{movie_id}_chunks.json
        data/temporal_chunks/all_chunks.json  (merged for FAISS re-indexing)

Usage:
    python scripts/build_temporal_chunks.py
    python scripts/build_temporal_chunks.py --movie tt0120338
    python scripts/build_temporal_chunks.py --fps 24
"""

import json
import re
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TemporalChunkBuilder")

#  Paths 
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
ANNOTATION_DIR = BASE_DIR / "movienet_subset" / "annotation"
SUBTITLE_DIR = BASE_DIR / "movienet_subset" / "subtitle"
META_DIR = BASE_DIR / "movienet_subset" / "meta"
UNIFIED_DATASET = BASE_DIR / "unified_dataset" / "movierag_dataset.json"
SHOT_KEYF_DIR = BASE_DIR / "movienet" / "shot_keyf"
RAW_VIDEOS_DIR = BASE_DIR / "raw_videos"
OUTPUT_DIR = BASE_DIR / "temporal_chunks"

# Fallback meta directories
META_DIRS = [
    BASE_DIR / "movienet_subset" / "meta",
    BASE_DIR / "unified_dataset" / "meta",
]


# 
# Layer ①: Annotation Frame Boundaries → Exact Timestamps
# 


def get_video_fps(movie_id: str) -> float:
    """Get FPS from the raw video file using OpenCV. Falls back to 24.0."""
    for ext in [".mp4", ".mkv", ".avi"]:
        vpath = RAW_VIDEOS_DIR / f"{movie_id}{ext}"
        if vpath.exists():
            try:
                import cv2

                cap = cv2.VideoCapture(str(vpath))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    logger.info(f"  [FPS] {movie_id}: {fps:.3f} FPS from video file")
                    return fps
            except Exception:
                pass
    logger.warning(f"  [FPS] {movie_id}: No video found, using default 24.0 FPS")
    return 24.0


def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_annotation_scenes(movie_id: str) -> List[Dict]:
    """
    Load scene boundaries from annotation JSON.
    Returns list of {scene_id, shot_range, frame_range, start_time, end_time}
    """
    ann_path = ANNOTATION_DIR / f"{movie_id}.json"
    if not ann_path.exists():
        return []

    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"  Failed to parse annotation for {movie_id}: {e}")
        return []

    scenes_raw = data.get("scene", [])
    if not scenes_raw:
        return []

    fps = get_video_fps(movie_id)

    scenes = []
    for s in scenes_raw:
        frame_range = s.get("frame", [0, 0])
        shot_range = s.get("shot", [0, 0])
        if len(frame_range) < 2 or len(shot_range) < 2:
            continue

        start_sec = frame_range[0] / fps
        end_sec = frame_range[1] / fps

        scenes.append(
            {
                "scene_id": s.get("id", ""),
                "shot_start": shot_range[0],
                "shot_end": shot_range[1],
                "frame_start": frame_range[0],
                "frame_end": frame_range[1],
                "start_time": fmt_time(start_sec),
                "end_time": fmt_time(end_sec),
                "start_seconds": round(start_sec, 2),
                "end_seconds": round(end_sec, 2),
                "duration_seconds": round(end_sec - start_sec, 2),
                "place_tag": s.get("place_tag"),
                "action_tag": s.get("action_tag"),
            }
        )

    logger.info(f"  [Layer①] {movie_id}: {len(scenes)} scenes with frame boundaries")
    return scenes


# 
# Layer ②: MovieGraphs Clip Semantics
# 


def load_clip_semantics(movie_id: str, unified_data: Dict) -> List[Dict]:
    """Load semantic clips from the unified movierag_dataset.json."""
    movie = unified_data.get("movies", {}).get(movie_id)
    if not movie:
        return []

    clips = movie.get("clips", [])
    result = []
    for clip in clips:
        result.append(
            {
                "clip_id": clip.get("clip_id", ""),
                "start_shot": clip.get("start_shot", 0),
                "end_shot": clip.get("end_shot", 0),
                "description": clip.get("description", ""),
                "situation": clip.get("situation", ""),
                "scene_label": clip.get("scene_label", ""),
                "characters": [c.get("name", "") for c in clip.get("characters", [])],
                "character_ids": [c.get("id", "") for c in clip.get("characters", [])],
                "entities": clip.get("entities", []),
                "attributes": list(set(clip.get("attributes", []))),  # deduplicate
                "interactions": clip.get("interactions", []),
            }
        )

    logger.info(f"  [Layer②] {movie_id}: {len(result)} semantic clips")
    return result


# 
# Layer ③: Subtitle/Dialogue Alignment
# 


def parse_srt(srt_path: Path) -> List[Dict]:
    """Parse SRT subtitle file into structured entries."""
    if not srt_path.exists():
        return []

    try:
        text = srt_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    entries = []
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Parse timestamp line: 00:01:16,820 --> 00:01:19,660
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1].strip(),
        )
        if not ts_match:
            continue

        def srt_to_sec(ts: str) -> float:
            ts = ts.replace(",", ".")
            parts = ts.split(":")
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

        start_sec = srt_to_sec(ts_match.group(1))
        end_sec = srt_to_sec(ts_match.group(2))
        dialogue = " ".join(lines[2:]).strip()
        # Remove HTML tags from subtitle
        dialogue = re.sub(r"<[^>]+>", "", dialogue)

        if dialogue:
            entries.append(
                {
                    "start_seconds": round(start_sec, 3),
                    "end_seconds": round(end_sec, 3),
                    "text": dialogue,
                }
            )

    return entries


def align_subtitles(
    srt_entries: List[Dict], start_sec: float, end_sec: float
) -> List[str]:
    """Find all subtitle entries that overlap with [start_sec, end_sec]."""
    dialogues = []
    for entry in srt_entries:
        # Check overlap: subtitle overlaps with scene if
        # subtitle_start < scene_end AND subtitle_end > scene_start
        if entry["start_seconds"] < end_sec and entry["end_seconds"] > start_sec:
            dialogues.append(entry["text"])
    return dialogues


# 
# Layer ④: Movie Metadata (Cast, Genre, Title)
# 


def load_movie_meta(movie_id: str) -> Dict:
    """Load movie metadata from meta JSON."""
    for d in META_DIRS:
        p = d / f"{movie_id}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


# 
# Layer ⑤: Representative Keyframe Paths (Time-based matching)
#
# Strategy:
#   A) If keyframe_index.json exists (from extract_scene_keyframes.py):
#      Use EXACT timestamps stored in the index — each keyframe has a
#      precise timestamp_sec recorded from ffprobe/annotation data.
#   B) Fallback: If no index, use old heuristic: shot_XXXX → XXXX * 3s
#      (for directories created by build_dataset_from_local_movies.py)
# 

# Fallback extraction interval (for old-style directories without index)
KEYFRAME_INTERVAL_SEC = 3.0

# Also search these directories for keyframes
KEYF_SEARCH_DIRS = [
    BASE_DIR / "movienet" / "shot_keyf",
    BASE_DIR / "Standalone_Dataset" / "shot_keyf",
]


def _build_keyframe_time_index(movie_id: str) -> List[Dict]:
    """
    Build a time-indexed list of keyframes for a movie.

    1. First tries to load keyframe_index.json (precise timestamps).
    2. Falls back to heuristic (shot_num * 3s) for old directories.

    Returns sorted list of {path, timestamp_sec, scene_idx, img_idx}.
    """
    # Find the keyframe directory
    keyf_dir = None
    for search_dir in KEYF_SEARCH_DIRS:
        d = search_dir / movie_id
        if d.exists() and any(d.glob("shot_*.jpg")):
            keyf_dir = d
            break

    if not keyf_dir:
        return []

    #  Strategy A: Use keyframe_index.json (precise) 
    index_path = keyf_dir / "keyframe_index.json"
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            kf_list = data.get("keyframes", [])
            result = []
            for kf in kf_list:
                result.append(
                    {
                        "path": kf["path"],
                        "timestamp_sec": kf["timestamp_sec"],
                        "scene_idx": kf.get("scene_idx", 0),
                        "img_idx": kf.get("img_idx", 0),
                        "scene_start_sec": kf.get("scene_start_sec", 0),
                        "scene_end_sec": kf.get("scene_end_sec", 0),
                        "source": "index_json",
                    }
                )
            logger.debug(
                f"  [Layer⑤] {movie_id}: Loaded {len(result)} keyframes from index JSON"
            )
            return result
        except Exception as e:
            logger.warning(f"  [Layer⑤] Failed to read index JSON: {e}")

    #  Strategy B: Fallback heuristic (shot_num * 3s) 
    result = []
    for img_path in sorted(keyf_dir.glob("shot_*_img_*.jpg")):
        match = re.match(r"shot_(\d+)_img_(\d+)\.jpg", img_path.name)
        if match:
            shot_num = int(match.group(1))
            img_idx = int(match.group(2))
            timestamp_sec = shot_num * KEYFRAME_INTERVAL_SEC
            result.append(
                {
                    "path": str(img_path),
                    "timestamp_sec": timestamp_sec,
                    "scene_idx": shot_num,
                    "img_idx": img_idx,
                    "scene_start_sec": timestamp_sec,
                    "scene_end_sec": timestamp_sec + KEYFRAME_INTERVAL_SEC,
                    "source": "heuristic_3s",
                }
            )

    logger.debug(f"  [Layer⑤] {movie_id}: Loaded {len(result)} keyframes via heuristic")
    return result


# Cache: movie_id → keyframe time index
_keyframe_index_cache: Dict[str, List[Dict]] = {}


def find_keyframes_by_time(
    movie_id: str, start_sec: float, end_sec: float
) -> List[str]:
    """
    Find keyframe image paths whose timestamp falls within [start_sec, end_sec].

    Uses keyframe_index.json for precise timestamps if available,
    else falls back to shot_num * 3s heuristic.
    """
    # Build or get cached index
    if movie_id not in _keyframe_index_cache:
        _keyframe_index_cache[movie_id] = _build_keyframe_time_index(movie_id)

    kf_index = _keyframe_index_cache[movie_id]
    if not kf_index:
        return []

    # If time range is invalid or zero, return empty
    if end_sec <= start_sec or (start_sec == 0 and end_sec == 0):
        return []

    # Find keyframes in the time range
    # Group by scene_idx to pick best img_idx per scene
    scene_groups: Dict[int, List[Dict]] = {}
    for kf in kf_index:
        if start_sec <= kf["timestamp_sec"] <= end_sec:
            sn = kf["scene_idx"]
            if sn not in scene_groups:
                scene_groups[sn] = []
            scene_groups[sn].append(kf)

    # Pick representative frame per scene (prefer img_1=middle, then img_0, then img_2)
    IMG_PRIORITY = {1: 0, 0: 1, 2: 2}  # lower = preferred
    paths = []
    for sn in sorted(scene_groups.keys()):
        candidates = sorted(
            scene_groups[sn], key=lambda x: IMG_PRIORITY.get(x["img_idx"], 99)
        )
        paths.append(candidates[0]["path"])

    return paths


# 
# Merge: Build Super Chunks
# 


def match_clip_to_scene(clip: Dict, scenes: List[Dict]) -> Optional[Dict]:
    """
    Find the best matching scene for a clip based on shot overlap.
    A clip has start_shot/end_shot, a scene has shot_start/shot_end.
    We find the scene with maximum overlap.
    """
    clip_start = clip["start_shot"]
    clip_end = clip["end_shot"]

    best_scene = None
    best_overlap = 0

    for scene in scenes:
        s_start = scene["shot_start"]
        s_end = scene["shot_end"]

        # Calculate overlap
        overlap_start = max(clip_start, s_start)
        overlap_end = min(clip_end, s_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_scene = scene

    # If no overlap found, find the scene containing the clip's start_shot
    if not best_scene:
        for scene in scenes:
            if scene["shot_start"] <= clip_start < scene["shot_end"]:
                best_scene = scene
                break

    return best_scene


def build_chunks_for_movie(
    movie_id: str,
    unified_data: Dict,
    default_fps: float = 24.0,
) -> List[Dict]:
    """Build all temporal chunks for a single movie."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Processing: {movie_id}")
    logger.info(f"{'=' * 60}")

    # Load all 5 layers
    scenes = load_annotation_scenes(movie_id)
    clips = load_clip_semantics(movie_id, unified_data)
    meta = load_movie_meta(movie_id)

    # Load subtitles
    srt_entries = []
    for ext in [".srt"]:
        srt_path = SUBTITLE_DIR / f"{movie_id}{ext}"
        if srt_path.exists():
            srt_entries = parse_srt(srt_path)
            logger.info(f"  [Layer③] {movie_id}: {len(srt_entries)} subtitle entries")
            break

    if not srt_entries:
        logger.info(f"  [Layer③] {movie_id}: No subtitle file found")

    # Movie context
    title = meta.get("title", movie_id)
    genres = meta.get("genres", [])
    cast_map = {}
    for c in meta.get("cast", []):
        name = c.get("name", "")
        character = c.get("character", "")
        if name:
            cast_map[name] = character

    logger.info(
        f"  [Layer④] {movie_id}: title='{title}', genres={genres}, cast={len(cast_map)}"
    )

    chunks = []
    chunk_id = 0

    if scenes and clips:
        #  Strategy A: Clip-centric with Scene timestamp anchoring 
        # Each MovieGraphs clip becomes a chunk, enriched with exact timestamps
        # from the matching annotation scene.
        for clip in clips:
            matched_scene = match_clip_to_scene(clip, scenes)

            if matched_scene:
                start_time = matched_scene["start_time"]
                end_time = matched_scene["end_time"]
                start_sec = matched_scene["start_seconds"]
                end_sec = matched_scene["end_seconds"]
                timestamp_source = "annotation_frame"
            else:
                # Fallback: no matching scene, skip or estimate
                start_sec = 0
                end_sec = 0
                start_time = "00:00:00"
                end_time = "00:00:00"
                timestamp_source = "none"

            # Get dialogue for this time range
            dialogues = (
                align_subtitles(srt_entries, start_sec, end_sec)
                if start_sec or end_sec
                else []
            )

            # Get keyframe paths (time-based matching: shot_XXXX → XXXX*3 seconds)
            keyframes = find_keyframes_by_time(movie_id, start_sec, end_sec)

            chunk = {
                "chunk_id": f"{movie_id}_chunk_{chunk_id:04d}",
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
                # Temporal (Layer ①)
                "start_time": start_time,
                "end_time": end_time,
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "duration_seconds": round(end_sec - start_sec, 2),
                "timestamp_source": timestamp_source,
                # Semantic (Layer ②)
                "clip_id": clip["clip_id"],
                "description": clip["description"],
                "situation": clip["situation"],
                "scene_label": clip["scene_label"],
                "characters": clip["characters"],
                "character_ids": clip["character_ids"],
                "attributes": clip["attributes"],
                "interactions": clip["interactions"],
                # Dialogue (Layer ③)
                "dialogue": dialogues,
                "dialogue_text": " ".join(dialogues) if dialogues else "",
                # Shot range
                "shot_start": clip["start_shot"],
                "shot_end": clip["end_shot"],
                # Keyframes (Layer ⑤)
                "keyframe_paths": keyframes,
                "num_keyframes": len(keyframes),
                # Graph hint (Layer ④ — actor mapping for Neo4j enrichment)
                "cast_in_scene": [
                    {"actor": actor, "character": char}
                    for char_name in clip["characters"]
                    for actor, char in cast_map.items()
                    if char_name.lower() in char.lower()
                    or char.lower() in char_name.lower()
                ],
            }
            chunks.append(chunk)
            chunk_id += 1

    elif scenes:
        #  Strategy B: Scene-only (no MovieGraphs clips) 
        # Use annotation scenes directly with subtitle alignment
        for scene in scenes:
            dialogues = align_subtitles(
                srt_entries, scene["start_seconds"], scene["end_seconds"]
            )
            keyframes = find_keyframes_by_time(
                movie_id, scene["start_seconds"], scene["end_seconds"]
            )

            chunk = {
                "chunk_id": f"{movie_id}_chunk_{chunk_id:04d}",
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "start_seconds": scene["start_seconds"],
                "end_seconds": scene["end_seconds"],
                "duration_seconds": scene["duration_seconds"],
                "timestamp_source": "annotation_frame",
                "clip_id": "",
                "description": "",
                "situation": "",
                "scene_label": scene.get("place_tag", "") or "",
                "characters": [],
                "character_ids": [],
                "attributes": [],
                "interactions": [],
                "dialogue": dialogues,
                "dialogue_text": " ".join(dialogues) if dialogues else "",
                "shot_start": scene["shot_start"],
                "shot_end": scene["shot_end"],
                "keyframe_paths": keyframes,
                "num_keyframes": len(keyframes),
                "cast_in_scene": [],
            }
            chunks.append(chunk)
            chunk_id += 1

    else:
        logger.warning(
            f"  {movie_id}: No annotation scenes AND no clips found. Skipping."
        )

    # Stats
    with_ts = sum(1 for c in chunks if c["timestamp_source"] == "annotation_frame")
    with_desc = sum(1 for c in chunks if c["description"])
    with_dialog = sum(1 for c in chunks if c["dialogue"])
    with_keyf = sum(1 for c in chunks if c["keyframe_paths"])

    logger.info(f"  [RESULT] {movie_id}: {len(chunks)} chunks built")
    logger.info(f"     With exact timestamps: {with_ts}")
    logger.info(f"     With descriptions:     {with_desc}")
    logger.info(f"     With dialogue:          {with_dialog}")
    logger.info(f"     With keyframes:         {with_keyf}")

    return chunks


# 
# Main
# 


def main():
    parser = argparse.ArgumentParser(description="Build 5-Layer Temporal Chunks")
    parser.add_argument(
        "--movie", type=str, help="Process single movie (e.g. tt0120338)"
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Default FPS fallback")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load unified dataset
    logger.info(f"Loading unified dataset from: {UNIFIED_DATASET}")
    if UNIFIED_DATASET.exists():
        unified_data = json.loads(UNIFIED_DATASET.read_text(encoding="utf-8"))
        num_movies_dataset = len(unified_data.get("movies", {}))
        logger.info(f"  Loaded {num_movies_dataset} movies from unified dataset")
    else:
        unified_data = {"movies": {}}
        logger.warning("  Unified dataset not found, proceeding with annotations only")

    # Determine which movies to process
    if args.movie:
        movie_ids = [args.movie]
    else:
        # Collect all movie IDs from annotation + unified dataset
        ann_ids = (
            {p.stem for p in ANNOTATION_DIR.glob("*.json")}
            if ANNOTATION_DIR.exists()
            else set()
        )
        dataset_ids = set(unified_data.get("movies", {}).keys())
        movie_ids = sorted(ann_ids | dataset_ids)

    logger.info(f"\nProcessing {len(movie_ids)} movies...")

    all_chunks = []
    stats = {
        "total_movies": 0,
        "total_chunks": 0,
        "with_timestamps": 0,
        "with_dialogue": 0,
    }

    for movie_id in movie_ids:
        chunks = build_chunks_for_movie(movie_id, unified_data, args.fps)

        if chunks:
            # Save per-movie file
            out_path = OUTPUT_DIR / f"{movie_id}_chunks.json"
            out_path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            all_chunks.extend(chunks)
            stats["total_movies"] += 1
            stats["total_chunks"] += len(chunks)
            stats["with_timestamps"] += sum(
                1 for c in chunks if c["timestamp_source"] == "annotation_frame"
            )
            stats["with_dialogue"] += sum(1 for c in chunks if c["dialogue"])

    # Save merged file
    merged_path = OUTPUT_DIR / "all_chunks.json"
    merged_data = {
        "metadata": {
            "total_movies": stats["total_movies"],
            "total_chunks": stats["total_chunks"],
            "chunks_with_exact_timestamps": stats["with_timestamps"],
            "chunks_with_dialogue": stats["with_dialogue"],
        },
        "chunks": all_chunks,
    }
    merged_path.write_text(
        json.dumps(merged_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"   BUILD COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total movies processed:         {stats['total_movies']}")
    logger.info(f"  Total chunks created:            {stats['total_chunks']}")
    logger.info(f"  Chunks with exact timestamps:    {stats['with_timestamps']}")
    logger.info(f"  Chunks with dialogue:            {stats['with_dialogue']}")
    logger.info(
        f"  Per-movie files:                 {OUTPUT_DIR}/<movie_id>_chunks.json"
    )
    logger.info(f"  Merged file:                     {merged_path}")
    logger.info(f"\n  Next step: Re-index FAISS using all_chunks.json")


if __name__ == "__main__":
    main()
