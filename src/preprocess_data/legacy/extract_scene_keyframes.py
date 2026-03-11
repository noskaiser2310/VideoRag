"""
 Precision Shot Keyframe Extractor

Extracts keyframes directly from raw video files at EXACT scene boundary
timestamps from MovieNet annotation JSON files.

Instead of the old fixed 1/3 FPS approach (which creates meaningless sequential
frame indices), this script:

1. Reads annotation/{movie_id}.json → scene[].frame boundaries
2. Converts frame numbers to exact timestamps via video FPS
3. For each scene: extracts 3 representative keyframes using ffmpeg:
   - img_0: at scene START (first frame of scene)
   - img_1: at scene MIDDLE (halfway point)
   - img_2: at scene END (last frame of scene)
4. Saves with EMBEDDED timestamp metadata in filename:
   shot_{scene_idx:04d}_img_{0/1/2}_t{timestamp}s.jpg

Output: data/movienet/shot_keyf/{movie_id}/shot_XXXX_img_Y.jpg
        data/movienet/shot_keyf/{movie_id}/keyframe_index.json  (timestamp lookup)

Usage:
    python scripts/extract_scene_keyframes.py
    python scripts/extract_scene_keyframes.py --movie tt0120338
    python scripts/extract_scene_keyframes.py --force  (overwrite existing)
"""

import json
import os
import subprocess
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SceneKeyframeExtractor")

#  Paths 
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
ANNOTATION_DIR = BASE_DIR / "movienet_subset" / "annotation"
RAW_VIDEOS_DIR = BASE_DIR / "raw_videos"
OUTPUT_KEYF_DIR = BASE_DIR / "movienet" / "shot_keyf"


def get_video_path(movie_id: str) -> Optional[Path]:
    """Find the raw video file for a movie."""
    for ext in [".mp4", ".mkv", ".avi", ".mov"]:
        p = RAW_VIDEOS_DIR / f"{movie_id}{ext}"
        if p.exists():
            return p
    return None


def get_video_info(video_path: Path) -> Dict:
    """Get video FPS and duration using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Find the video stream
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                # Parse FPS from r_frame_rate (e.g., "24000/1001")
                fps_str = stream.get("r_frame_rate", "24/1")
                parts = fps_str.split("/")
                fps = (
                    float(parts[0]) / float(parts[1])
                    if len(parts) == 2
                    else float(parts[0])
                )

                duration = float(data.get("format", {}).get("duration", 0))

                return {
                    "fps": fps,
                    "duration": duration,
                    "width": int(stream.get("width", 0)),
                    "height": int(stream.get("height", 0)),
                }
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}, using default FPS=24")

    return {"fps": 24.0, "duration": 0, "width": 0, "height": 0}


def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_time_ffmpeg(seconds: float) -> str:
    """Format seconds for ffmpeg (HH:MM:SS.mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def load_scenes(movie_id: str, fps: float) -> List[Dict]:
    """Load and convert annotation scenes to timestamp-based entries."""
    ann_path = ANNOTATION_DIR / f"{movie_id}.json"
    if not ann_path.exists():
        return []

    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to parse annotation: {e}")
        return []

    scenes_raw = data.get("scene", [])
    if not scenes_raw:
        return []

    scenes = []
    for i, s in enumerate(scenes_raw):
        frame_range = s.get("frame", [0, 0])
        if len(frame_range) < 2:
            continue

        start_sec = frame_range[0] / fps
        end_sec = frame_range[1] / fps

        # Skip tiny scenes (< 0.5s)
        if end_sec - start_sec < 0.5:
            continue

        mid_sec = (start_sec + end_sec) / 2

        scenes.append(
            {
                "scene_idx": i,
                "scene_id": s.get("id", f"scene_{i}"),
                "frame_start": frame_range[0],
                "frame_end": frame_range[1],
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "mid_sec": round(mid_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
            }
        )

    return scenes


def extract_frame_at_time(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path,
    height: int = 720,
) -> bool:
    """Extract a single frame from video at exact timestamp using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-ss",
        fmt_time_ffmpeg(timestamp_sec),  # Seek to exact position
        "-i",
        str(video_path),
        "-vframes",
        "1",  # Extract exactly 1 frame
        "-vf",
        f"scale=-1:{height}",  # Resize to target height
        "-qscale:v",
        "2",  # High quality JPEG
        "-y",  # Overwrite
        "-v",
        "quiet",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, timeout=30)
        return output_path.exists()
    except Exception:
        return False


def process_movie(movie_id: str, force: bool = False) -> Dict:
    """Process a single movie: extract keyframes at scene boundaries."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"   Processing: {movie_id}")
    logger.info(f"{'=' * 60}")

    # 1. Check video exists
    video_path = get_video_path(movie_id)
    if not video_path:
        logger.warning(f"   No video file found for {movie_id}")
        return {"movie_id": movie_id, "status": "no_video", "scenes": 0, "keyframes": 0}

    # 2. Get video info
    info = get_video_info(video_path)
    fps = info["fps"]
    duration = info["duration"]
    logger.info(
        f"   Video: {video_path.name} | FPS: {fps:.3f} | Duration: {fmt_time(duration)}"
    )

    # 3. Load annotation scenes
    scenes = load_scenes(movie_id, fps)
    if not scenes:
        logger.warning(f"   No valid scenes in annotation for {movie_id}")
        return {
            "movie_id": movie_id,
            "status": "no_scenes",
            "scenes": 0,
            "keyframes": 0,
        }

    logger.info(f"   Found {len(scenes)} scenes to extract")

    # 4. Prepare output directory
    out_dir = OUTPUT_KEYF_DIR / movie_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if not force:
        existing = list(out_dir.glob("shot_*.jpg"))
        if len(existing) >= len(scenes) * 3:
            logger.info(
                f"   Already processed ({len(existing)} keyframes). Use --force to overwrite."
            )
            return {
                "movie_id": movie_id,
                "status": "skipped",
                "scenes": len(scenes),
                "keyframes": len(existing),
            }

    # 5. Extract 3 keyframes per scene
    keyframe_index = []
    extracted = 0
    t_start = time.time()

    for scene in scenes:
        idx = scene["scene_idx"]

        # 3 extraction points: start, middle, end
        timestamps = [
            (0, scene["start_sec"]),  # img_0: scene start
            (1, scene["mid_sec"]),  # img_1: scene middle
            (
                2,
                max(scene["start_sec"], scene["end_sec"] - 0.1),
            ),  # img_2: scene end (slightly before boundary)
        ]

        for img_idx, ts in timestamps:
            # Ensure timestamp is within video bounds
            ts = max(0, min(ts, duration - 0.1))

            fname = f"shot_{idx:04d}_img_{img_idx}.jpg"
            out_path = out_dir / fname

            if extract_frame_at_time(video_path, ts, out_path):
                extracted += 1
                keyframe_index.append(
                    {
                        "filename": fname,
                        "scene_idx": idx,
                        "scene_id": scene["scene_id"],
                        "img_idx": img_idx,
                        "timestamp_sec": round(ts, 3),
                        "timestamp_fmt": fmt_time(ts),
                        "scene_start_sec": scene["start_sec"],
                        "scene_end_sec": scene["end_sec"],
                        "path": str(out_path),
                    }
                )

    elapsed = time.time() - t_start

    # 6. Save keyframe index JSON (timestamp lookup table)
    index_path = out_dir / "keyframe_index.json"
    index_data = {
        "movie_id": movie_id,
        "video_fps": fps,
        "video_duration": duration,
        "total_scenes": len(scenes),
        "total_keyframes": extracted,
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keyframes": keyframe_index,
    }
    index_path.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"   Extracted {extracted} keyframes in {elapsed:.1f}s")
    logger.info(f"   Saved to: {out_dir}")
    logger.info(f"   Index: {index_path}")

    return {
        "movie_id": movie_id,
        "status": "ok",
        "scenes": len(scenes),
        "keyframes": extracted,
        "elapsed": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract Scene Keyframes from Raw Videos"
    )
    parser.add_argument(
        "--movie", type=str, help="Process single movie (e.g. tt0120338)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing keyframes"
    )
    args = parser.parse_args()

    logger.info(" PRECISION SCENE KEYFRAME EXTRACTOR")
    logger.info("=" * 60)

    # Determine which movies to process
    if args.movie:
        movie_ids = [args.movie]
    else:
        # Find all movies that have BOTH annotation + raw video
        ann_ids = (
            {p.stem for p in ANNOTATION_DIR.glob("*.json")}
            if ANNOTATION_DIR.exists()
            else set()
        )
        video_ids = (
            {p.stem for p in RAW_VIDEOS_DIR.glob("*.*")}
            if RAW_VIDEOS_DIR.exists()
            else set()
        )
        movie_ids = sorted(ann_ids & video_ids)

    logger.info(f"Movies to process: {len(movie_ids)}")

    results = []
    total_start = time.time()

    for i, mid in enumerate(movie_ids, 1):
        logger.info(f"\n[{i}/{len(movie_ids)}]")
        result = process_movie(mid, force=args.force)
        results.append(result)

    total_elapsed = time.time() - total_start

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] not in ("ok", "skipped")]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"   EXTRACTION COMPLETE ({total_elapsed:.1f}s)")
    logger.info(f"{'=' * 60}")
    logger.info(
        f"  Extracted:  {len(ok)} movies, {sum(r['keyframes'] for r in ok)} keyframes"
    )
    logger.info(f"  Skipped:    {len(skipped)} (already processed)")
    logger.info(f"  Failed:     {len(failed)}")
    logger.info(f"\n  Output: {OUTPUT_KEYF_DIR}")
    logger.info(f"\n  Each movie now has:")
    logger.info(f"    shot_XXXX_img_0.jpg  (scene START)")
    logger.info(f"    shot_XXXX_img_1.jpg  (scene MIDDLE)")
    logger.info(f"    shot_XXXX_img_2.jpg  (scene END)")
    logger.info(f"    keyframe_index.json  (timestamp lookup)")
    logger.info(f"\n  Next: Run build_temporal_chunks.py then reindex_temporal.py")


if __name__ == "__main__":
    main()
