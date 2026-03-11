"""
Metadata Crawler — IMDB/TMDB → meta JSON

Wraps movienet_tools crawler to fetch movie metadata from online sources
and save as meta JSON compatible with the existing pipeline.

For new videos when IMDB ID is known.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class MetadataCrawler:
    """Fetch movie metadata from IMDB/TMDB and save as JSON."""

    def __init__(self):
        self._imdb_crawler = None
        self._tmdb_crawler = None
        self._init_crawlers()

    def _init_crawlers(self):
        """Initialize movienet crawlers if available."""
        try:
            crawler_path = Cfg.DATA_DIR / "movienet_tools"
            if crawler_path.exists():
                sys.path.insert(0, str(crawler_path))
                from movienet.tools.crawler.imdb_crawler import IMDBCrawler

                self._imdb_crawler = IMDBCrawler()
                logger.debug("  IMDB crawler initialized")
        except Exception as e:
            logger.debug(f"  IMDB crawler not available: {e}")

        try:
            from movienet.tools.crawler.tmdb_crawler import TMDBCrawler

            self._tmdb_crawler = TMDBCrawler()
            logger.debug("  TMDB crawler initialized")
        except Exception:
            pass

    def crawl(self, movie_id: str, force: bool = False) -> Dict:
        """
        Crawl metadata for a movie and save as JSON.

        Args:
            movie_id: IMDB ID (e.g., 'tt0120338') or custom ID
            force: Overwrite existing meta file

        Returns: metadata dict
        """
        # Check existing
        Cfg.META_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = Cfg.META_DIR / f"{movie_id}.json"

        if meta_path.exists() and not force:
            logger.info(f"   Meta already exists: {meta_path}")
            return json.loads(meta_path.read_text(encoding="utf-8"))

        logger.info(f"   Crawling metadata: {movie_id}")

        meta = {"imdb_id": movie_id}

        # IMDB crawling (requires tt-format ID)
        if movie_id.startswith("tt") and self._imdb_crawler:
            meta = self._crawl_imdb(movie_id)
        else:
            logger.info(f"  Non-IMDB ID or crawler unavailable, creating minimal meta")
            meta = {
                "imdb_id": movie_id,
                "title": movie_id.replace("_", " ").title(),
                "genres": [],
                "cast": [],
                "auto_generated": True,
            }

        # Save
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"   Meta saved: {meta.get('title', movie_id)} → {meta_path}")
        return meta

    def _crawl_imdb(self, imdb_id: str) -> Dict:
        """Crawl all available data from IMDB."""
        meta = {"imdb_id": imdb_id, "auto_generated": True}

        # Home page: title, genres, storyline, country
        try:
            home = self._imdb_crawler.parse_home_page(imdb_id)
            meta.update(
                {
                    "title": home.get("title", imdb_id),
                    "genres": home.get("genres", []),
                    "country": home.get("country"),
                    "storyline": home.get("storyline"),
                }
            )
            logger.info(f"    Title: {meta.get('title')}")
        except Exception as e:
            logger.warning(f"    IMDB home page failed: {e}")

        # Credits page: director, cast
        try:
            credits = self._imdb_crawler.parse_credits_page(imdb_id)
            meta["director"] = credits.get("director")
            meta["cast"] = credits.get("cast", [])
            logger.info(f"    Cast: {len(meta.get('cast', []))} actors")
        except Exception as e:
            logger.warning(f"    IMDB credits page failed: {e}")

        # Synopsis
        try:
            synopsis = self._imdb_crawler.parse_synopsis(imdb_id)
            if synopsis.get("synopsis"):
                meta["synopsis"] = synopsis["synopsis"]
        except Exception as e:
            logger.debug(f"    IMDB synopsis failed: {e}")

        return meta
