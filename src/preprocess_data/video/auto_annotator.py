"""
Auto Annotator — Shot Detection → Annotation JSON

Uses movienet_tools shotdetect (ContentDetectorHSVLUV) or ffmpeg I-frame
fallback to detect shot boundaries and generate annotation JSON equivalent
to MovieNet's annotation format.

For new videos without any pre-existing annotation data.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class AutoAnnotator:
    """Generate annotation JSON from shot detection on raw video."""

    def __init__(self):
        self._shotdetect_available = self._check_shotdetect()

    @staticmethod
    def _check_shotdetect():
        """Check if movienet shotdetect module is available."""
        try:
            shotdetect_path = (
                Cfg.DATA_DIR / "movienet_tools" / "movienet" / "tools" / "shotdetect"
            )
            if shotdetect_path.exists():
                sys.path.insert(0, str(Cfg.DATA_DIR / "movienet_tools"))
                from movienet.tools.shotdetect.shotdetector import ShotDetector

                return True
        except ImportError:
            pass
        return False

    def annotate(self, movie_id: str, video_path: Path = None) -> Dict:
        """
        Detect shots and generate annotation JSON for a movie.

        Returns: annotation dict compatible with MovieNet format
        """
        if video_path is None:
            video_path = Cfg.get_video_path(movie_id)
        if video_path is None or not video_path.exists():
            logger.error(f"   No video for {movie_id}")
            return {}

        logger.info(f"   Auto-annotating: {movie_id}")

        # Get video info
        from ..video.keyframe_extractor import KeyframeExtractor

        info = KeyframeExtractor.get_video_info(video_path)
        fps = info["fps"]
        duration = info["duration"]

        # Detect shots
        if self._shotdetect_available:
            shots = self._detect_with_movienet(movie_id, video_path)
        else:
            shots = self._detect_with_ffmpeg(video_path, fps)

        if not shots:
            logger.warning(f"  No shots detected, using fixed interval")
            shots = self._fixed_interval_shots(duration, fps)

        # Group shots into scenes (every 5-10 shots → 1 scene)
        scenes = self._group_shots_into_scenes(shots, fps)

        # Build annotation JSON
        annotation = {
            "movie_id": movie_id,
            "fps": fps,
            "duration": duration,
            "total_shots": len(shots),
            "total_scenes": len(scenes),
            "auto_generated": True,
            "scene": scenes,
        }

        # Save
        Cfg.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        ann_path = Cfg.ANNOTATION_DIR / f"{movie_id}.json"
        ann_path.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            f"   Auto-annotation: {len(shots)} shots → {len(scenes)} scenes → {ann_path}"
        )
        return annotation

    def _detect_with_movienet(self, movie_id: str, video_path: Path) -> List[Dict]:
        """Use movienet's ShotDetector (ContentDetectorHSVLUV)."""
        logger.info("  Using MovieNet ShotDetector (HSV+LUV)...")
        try:
            from movienet.tools.shotdetect.shotdetector import ShotDetector

            out_dir = str(Cfg.MOVIENET_DIR)
            sdt = ShotDetector(
                save_keyf=False,
                save_keyf_txt=True,
                begin_frame=0,
                end_frame=999999999,
                show_progress=True,
            )
            sdt.shotdetect(str(video_path), out_dir)

            # Read shot_txt output
            txt_path = Cfg.MOVIENET_DIR / "shot_txt" / f"{video_path.stem}.txt"
            return self._parse_shot_txt(txt_path)
        except Exception as e:
            logger.warning(f"  MovieNet shotdetect failed: {e}, falling back to ffmpeg")
            from ..video.keyframe_extractor import KeyframeExtractor

            info = KeyframeExtractor.get_video_info(video_path)
            return self._detect_with_ffmpeg(video_path, info["fps"])

    def _detect_with_ffmpeg(self, video_path: Path, fps: float) -> List[Dict]:
        """Fallback: detect I-frames / scene changes via ffmpeg."""
        logger.info("  Using ffmpeg scene detection...")
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v",
                "-show_frames",
                "-show_entries",
                "frame=pict_type,pts_time",
                "-of",
                "json",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            data = json.loads(result.stdout)

            # Extract I-frame timestamps
            i_frames = []
            for frame in data.get("frames", []):
                if frame.get("pict_type") == "I":
                    pts = float(frame.get("pts_time", 0))
                    i_frames.append(pts)

            if len(i_frames) < 3:
                return []

            # Convert I-frames to shot boundaries
            shots = []
            for i in range(len(i_frames) - 1):
                start_t = i_frames[i]
                end_t = i_frames[i + 1]
                # Filter: minimum shot duration 1 second
                if end_t - start_t >= 1.0:
                    shots.append(
                        {
                            "shot_idx": len(shots),
                            "start_frame": int(start_t * fps),
                            "end_frame": int(end_t * fps),
                            "start_sec": round(start_t, 3),
                            "end_sec": round(end_t, 3),
                        }
                    )
            return shots

        except Exception as e:
            logger.warning(f"  ffmpeg scene detection failed: {e}")
            return []

    @staticmethod
    def _parse_shot_txt(txt_path: Path) -> List[Dict]:
        """Parse movienet shot_txt output format."""
        if not txt_path.exists():
            return []
        shots = []
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                shots.append(
                    {
                        "shot_idx": len(shots),
                        "start_frame": int(parts[0]),
                        "end_frame": int(parts[1]),
                    }
                )
        return shots

    @staticmethod
    def _fixed_interval_shots(
        duration: float, fps: float, interval: float = 5.0
    ) -> List[Dict]:
        """Fallback: create shots at fixed intervals."""
        shots = []
        t = 0.0
        while t < duration:
            end_t = min(t + interval, duration)
            shots.append(
                {
                    "shot_idx": len(shots),
                    "start_frame": int(t * fps),
                    "end_frame": int(end_t * fps),
                    "start_sec": round(t, 3),
                    "end_sec": round(end_t, 3),
                }
            )
            t += interval
        return shots

    @staticmethod
    def _group_shots_into_scenes(
        shots: List[Dict], fps: float, max_shots_per_scene: int = 8
    ) -> List[Dict]:
        """Group consecutive shots into scenes."""
        if not shots:
            return []

        scenes = []
        current_shots = []

        for shot in shots:
            current_shots.append(shot)
            if len(current_shots) >= max_shots_per_scene:
                scenes.append(_make_scene(scenes, current_shots, fps))
                current_shots = []

        if current_shots:
            scenes.append(_make_scene(scenes, current_shots, fps))

        return scenes


def _make_scene(existing_scenes, shot_group, fps):
    """Create a scene dict from a group of shots."""
    idx = len(existing_scenes)
    first, last = shot_group[0], shot_group[-1]
    start_frame = first.get("start_frame", 0)
    end_frame = last.get("end_frame", 0)
    return {
        "id": f"scene_{idx}",
        "shot": [first["shot_idx"], last["shot_idx"]],
        "frame": [start_frame, end_frame],
        "start_seconds": first.get("start_sec", round(start_frame / fps, 3)),
        "end_seconds": last.get("end_sec", round(end_frame / fps, 3)),
    }
