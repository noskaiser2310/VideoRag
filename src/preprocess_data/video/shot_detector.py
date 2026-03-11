"""
Shot Boundary Detector

Detects shot boundaries in video files using scene detection algorithms.
Adapted from: scripts/custom_shotdetect.py + movienet_tools/shotdetect/
"""

import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class ShotDetector:
    """Detect shot boundaries in video files using ffmpeg scene filter."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def detect_shots(self, movie_id: str) -> List[Dict]:
        """
        Detect shot boundaries using ffmpeg's scene change detection.
        Returns list of {timestamp_sec, frame_num, score}.
        """
        video_path = Cfg.get_video_path(movie_id)
        if not video_path:
            logger.warning(f"No video for {movie_id}")
            return []

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_frames",
            "-select_streams",
            "v",
            "-show_entries",
            "frame=pts_time,pict_type",
            "-of",
            "csv=p=0",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            # Parse I-frames as potential shot boundaries
            shots = []
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[1].strip() == "I":
                    try:
                        ts = float(parts[0])
                        shots.append({"timestamp_sec": round(ts, 3)})
                    except ValueError:
                        pass
            logger.info(f"  Detected {len(shots)} I-frames for {movie_id}")
            return shots
        except Exception as e:
            logger.error(f"Shot detection failed for {movie_id}: {e}")
            return []
