"""
Subtitle (SRT) Parser

Parses .srt subtitle files and aligns dialogue to time ranges.
Extracted from: scripts/build_temporal_chunks.py
"""

import re
import logging
from pathlib import Path
from typing import List, Dict

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class SubtitleParser:
    """Parse and align SRT subtitles."""

    @staticmethod
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

            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                lines[1].strip(),
            )
            if not ts_match:
                continue

            start_sec = _srt_to_sec(ts_match.group(1))
            end_sec = _srt_to_sec(ts_match.group(2))
            dialogue = " ".join(lines[2:]).strip()
            dialogue = re.sub(r"<[^>]+>", "", dialogue)  # Remove HTML tags

            if dialogue:
                entries.append(
                    {
                        "start_seconds": round(start_sec, 3),
                        "end_seconds": round(end_sec, 3),
                        "text": dialogue,
                    }
                )

        return entries

    @staticmethod
    def load_for_movie(movie_id: str) -> List[Dict]:
        """Load subtitle entries for a movie from known directories."""
        for ext in [".srt"]:
            srt_path = Cfg.SUBTITLE_DIR / f"{movie_id}{ext}"
            if srt_path.exists():
                entries = SubtitleParser.parse_srt(srt_path)
                logger.info(f"  [Subtitle] {movie_id}: {len(entries)} entries")
                return entries
        return []

    @staticmethod
    def align(srt_entries: List[Dict], start_sec: float, end_sec: float) -> List[str]:
        """Find all subtitle entries overlapping with [start_sec, end_sec]."""
        return [
            e["text"]
            for e in srt_entries
            if e["start_seconds"] < end_sec and e["end_seconds"] > start_sec
        ]


def _srt_to_sec(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
