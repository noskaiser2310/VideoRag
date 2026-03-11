"""
Precision Scene Keyframe Extractor

Extracts keyframes directly from raw video files at exact scene boundary
timestamps from MovieNet annotation JSON files.

Adapted from: scripts/extract_scene_keyframes.py
"""

import json
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """Extract keyframes at exact annotation scene boundaries."""

    def __init__(self, height: int = None, quality: int = None):
        self.height = height or Cfg.KEYFRAME_HEIGHT
        self.quality = quality or Cfg.KEYFRAME_QUALITY

    #  Video Info 

    @staticmethod
    def get_video_info(video_path: Path) -> Dict:
        """Get FPS and duration via ffprobe."""
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

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
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
            logger.warning(f"ffprobe failed: {e}")
        return {"fps": 24.0, "duration": 0, "width": 0, "height": 0}

    #  Scene Loading 

    @staticmethod
    def load_scenes(movie_id: str, fps: float) -> List[Dict]:
        """Load annotation scenes and convert to timestamp-based entries."""
        ann_path = Cfg.ANNOTATION_DIR / f"{movie_id}.json"
        if not ann_path.exists():
            return []

        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse annotation: {e}")
            return []

        scenes = []
        for i, s in enumerate(data.get("scene", [])):
            frame_range = s.get("frame", [0, 0])
            if len(frame_range) < 2:
                continue

            start_sec = frame_range[0] / fps
            end_sec = frame_range[1] / fps
            if end_sec - start_sec < 0.5:
                continue

            scenes.append(
                {
                    "scene_idx": i,
                    "scene_id": s.get("id", f"scene_{i}"),
                    "frame_start": frame_range[0],
                    "frame_end": frame_range[1],
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "mid_sec": round((start_sec + end_sec) / 2, 3),
                    "duration_sec": round(end_sec - start_sec, 3),
                }
            )
        return scenes

    #  Single Frame Extraction 

    def extract_frame_at_time(
        self, video_path: Path, timestamp_sec: float, output_path: Path
    ) -> bool:
        """Extract a single frame from video at exact timestamp."""
        h, m, s = (
            int(timestamp_sec // 3600),
            int((timestamp_sec % 3600) // 60),
            timestamp_sec % 60,
        )
        ts_str = f"{h:02d}:{m:02d}:{s:06.3f}"

        cmd = [
            "ffmpeg",
            "-ss",
            ts_str,
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale=-1:{self.height}",
            "-qscale:v",
            str(self.quality),
            "-y",
            "-v",
            "quiet",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=30)
            return output_path.exists()
        except Exception:
            return False

    #  Movie Processing 

    def process_movie(self, movie_id: str, force: bool = False) -> Dict:
        """Extract 3 keyframes per scene (start/middle/end) for a movie."""
        video_path = Cfg.get_video_path(movie_id)
        if not video_path:
            logger.warning(f"   No video for {movie_id}")
            return {"movie_id": movie_id, "status": "no_video", "keyframes": 0}

        info = self.get_video_info(video_path)
        fps, duration = info["fps"], info["duration"]
        logger.info(
            f"   {video_path.name} | FPS: {fps:.3f} | Duration: {_fmt(duration)}"
        )

        scenes = self.load_scenes(movie_id, fps)
        if not scenes:
            return {"movie_id": movie_id, "status": "no_scenes", "keyframes": 0}

        out_dir = Cfg.SHOT_KEYF_DIR / movie_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if not force:
            existing = list(out_dir.glob("shot_*.jpg"))
            if len(existing) >= len(scenes) * 3:
                logger.info(
                    f"   Already done ({len(existing)} keyframes). Use --force."
                )
                return {
                    "movie_id": movie_id,
                    "status": "skipped",
                    "keyframes": len(existing),
                }

        logger.info(f"   Extracting from {len(scenes)} scenes...")
        keyframe_index = []
        extracted = 0
        t0 = time.time()

        for scene in scenes:
            idx = scene["scene_idx"]
            timestamps = [
                (0, scene["start_sec"]),
                (1, scene["mid_sec"]),
                (2, max(scene["start_sec"], scene["end_sec"] - 0.1)),
            ]
            for img_idx, ts in timestamps:
                ts = max(0, min(ts, duration - 0.1))
                fname = f"shot_{idx:04d}_img_{img_idx}.jpg"
                out_path = out_dir / fname

                if self.extract_frame_at_time(video_path, ts, out_path):
                    extracted += 1
                    keyframe_index.append(
                        {
                            "filename": fname,
                            "scene_idx": idx,
                            "scene_id": scene["scene_id"],
                            "img_idx": img_idx,
                            "timestamp_sec": round(ts, 3),
                            "timestamp_fmt": _fmt(ts),
                            "scene_start_sec": scene["start_sec"],
                            "scene_end_sec": scene["end_sec"],
                            "path": str(out_path),
                        }
                    )

        # Save index
        index_data = {
            "movie_id": movie_id,
            "video_fps": fps,
            "video_duration": duration,
            "total_scenes": len(scenes),
            "total_keyframes": extracted,
            "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keyframes": keyframe_index,
        }
        (out_dir / "keyframe_index.json").write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        elapsed = time.time() - t0
        logger.info(f"   {extracted} keyframes in {elapsed:.1f}s → {out_dir}")
        return {
            "movie_id": movie_id,
            "status": "ok",
            "keyframes": extracted,
            "elapsed": round(elapsed, 1),
        }

    def process_all(
        self, movie_ids: List[str] = None, force: bool = False
    ) -> List[Dict]:
        """Process multiple movies."""
        ids = movie_ids or self._get_processable_ids()
        logger.info(f"Processing {len(ids)} movies...")
        results = []
        for i, mid in enumerate(ids, 1):
            logger.info(f"\n[{i}/{len(ids)}] {mid}")
            results.append(self.process_movie(mid, force=force))
        return results

    @staticmethod
    def _get_processable_ids() -> List[str]:
        """Movies that have both annotation AND raw video."""
        ann_ids = (
            {p.stem for p in Cfg.ANNOTATION_DIR.glob("*.json")}
            if Cfg.ANNOTATION_DIR.exists()
            else set()
        )
        vid_ids = set()
        for d in [Cfg.RAW_VIDEOS_DIR, Cfg.RAW_MOVIES_DIR]:
            if d.exists():
                vid_ids |= {
                    p.stem
                    for p in d.glob("*.*")
                    if p.suffix in {".mp4", ".mkv", ".avi"}
                }
        return sorted(ann_ids & vid_ids)


def _fmt(seconds: float) -> str:
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
