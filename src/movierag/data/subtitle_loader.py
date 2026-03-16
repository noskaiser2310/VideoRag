"""
Subtitle Loader for MovieRAG.

Parses SRT subtitle files and provides timestamp-aligned dialog
for RAG indexing and temporal grounding.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str


def _parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def _clean_text(text: str) -> str:
    """Remove HTML tags and clean subtitle text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip()
    return text


class SubtitleLoader:
    """
    Loads and parses SRT subtitle files.

    Usage:
        loader = SubtitleLoader(subtitle_dir="movie_data_subset_20/subtitle")
        entries = loader.load("tt0097576")
        dialog = loader.get_dialog_for_timerange("tt0097576", 60.0, 120.0)
    """

    def __init__(self, subtitle_dir: str):
        self.subtitle_dir = Path(subtitle_dir)

    def get_available_movies(self) -> List[str]:
        """Get list of movie IDs with subtitle files."""
        if not self.subtitle_dir.exists():
            return []
        return [f.stem for f in sorted(self.subtitle_dir.glob("*.srt"))]

    def load(self, movie_id: str) -> List[SubtitleEntry]:
        """
        Parse an SRT file into a list of SubtitleEntry objects.

        Args:
            movie_id: IMDB ID (e.g., 'tt0097576')

        Returns:
            List of SubtitleEntry with timestamps and cleaned text
        """
        srt_path = self.subtitle_dir / f"{movie_id}.srt"
        if not srt_path.exists():
            logger.warning(f"No subtitle file found: {srt_path}")
            return []

        entries = []
        try:
            content = srt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read {srt_path}: {e}")
            return []

        # Split by double newline (SRT blocks)
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            # Line 1: index
            try:
                idx = int(lines[0].strip())
            except ValueError:
                continue

            # Line 2: timestamps
            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                lines[1].strip(),
            )
            if not ts_match:
                continue

            start_sec = _parse_timestamp(ts_match.group(1))
            end_sec = _parse_timestamp(ts_match.group(2))

            # Lines 3+: text
            text = " ".join(
                _clean_text(line) for line in lines[2:] if _clean_text(line)
            )
            if not text:
                continue

            entries.append(
                SubtitleEntry(
                    index=idx,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    text=text,
                )
            )

        logger.info(f"Loaded {len(entries)} subtitle entries for {movie_id}")
        return entries

    def get_dialog_for_timerange(
        self, movie_id: str, start_sec: float, end_sec: float
    ) -> List[SubtitleEntry]:
        """
        Get subtitle entries overlapping with the given time range.

        Args:
            movie_id: IMDB ID
            start_sec: Start time in seconds
            end_sec: End time in seconds

        Returns:
            List of SubtitleEntry objects in the time range
        """
        entries = self.load(movie_id)
        return [
            e
            for e in entries
            if e.end_seconds >= start_sec and e.start_seconds <= end_sec
        ]

    def get_dialog_text_for_timerange(
        self, movie_id: str, start_sec: float, end_sec: float
    ) -> str:
        """Get concatenated dialog text for a time range."""
        entries = self.get_dialog_for_timerange(movie_id, start_sec, end_sec)
        return " ".join(e.text for e in entries)

    def get_textual_documents(
        self, movie_id: str, chunk_size: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Convert subtitles into chunked textual documents for RAG indexing.

        Groups subtitle entries into chunks of `chunk_size` entries,
        each forming one searchable document.

        Args:
            movie_id: IMDB ID
            chunk_size: Number of subtitle entries per chunk

        Returns:
            List of document dicts ready for KnowledgeIndexer
        """
        entries = self.load(movie_id)
        if not entries:
            return []

        documents = []
        for i in range(0, len(entries), chunk_size):
            chunk = entries[i : i + chunk_size]
            text_lines = [e.text for e in chunk]
            start_time = chunk[0].start_seconds
            end_time = chunk[-1].end_seconds

            def fmt(sec):
                m, s = divmod(int(sec), 60)
                h, m = divmod(m, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            full_text = (
                f"Dialog from {fmt(start_time)} to {fmt(end_time)}:\n"
                + " ".join(text_lines)
            )

            documents.append(
                {
                    "movie_id": movie_id,
                    "clip_id": f"subtitle_chunk_{i}",
                    "text": full_text,
                    "metadata": {
                        "category": "subtitle",
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_time_fmt": fmt(start_time),
                        "end_time_fmt": fmt(end_time),
                        "entry_count": len(chunk),
                    },
                }
            )

        return documents

    def get_all_textual_documents(self, chunk_size: int = 30) -> List[Dict[str, Any]]:
        """Get textual documents for all available movies."""
        all_docs = []
        for movie_id in self.get_available_movies():
            docs = self.get_textual_documents(movie_id, chunk_size)
            all_docs.extend(docs)
        logger.info(
            f"Generated {len(all_docs)} subtitle documents "
            f"from {len(self.get_available_movies())} movies"
        )
        return all_docs
