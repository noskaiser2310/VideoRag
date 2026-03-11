"""
Download Real Full-Length Movies for MovieRAG.
Uses yt-dlp to search YouTube for complete movies (60+ minutes).
Maps IMDb IDs to titles via cinemagoer, then downloads the best match.
"""

import os
import sys
import json
import time
from pathlib import Path

try:
    import yt_dlp
    from imdb import Cinemagoer

    DEPENDENCIES_OK = True
except ImportError:
    DEPENDENCIES_OK = False

# Configuration
DVDS_TXT_PATH = Path(r"D:\Study\School\project_ky4\data\MovieGraphs_repo\dvds.txt")
DOWNLOAD_DIR = Path(r"D:\Study\School\project_ky4\data\raw_videos")
PROGRESS_FILE = DOWNLOAD_DIR / "_download_progress.json"

# Minimum duration: 60 minutes = 3600 seconds
MIN_DURATION_SECONDS = 3600


def get_imdb_ids(txt_path, limit=None):
    """Extracts IMDb IDs from the dvds.txt file."""
    movie_ids = []
    if not txt_path.exists():
        print(f"File not found: {txt_path}")
        return movie_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("tt"):
                    movie_ids.append(parts[1])
                    if limit and len(movie_ids) >= limit:
                        break
    return movie_ids


def get_movie_title(ia, imdb_id):
    """Fetches movie title and year from IMDb."""
    try:
        numeric_id = imdb_id[2:]
        movie = ia.get_movie(numeric_id)
        title = movie.get("title")
        year = movie.get("year")
        return title, year
    except Exception as e:
        print(f"  [WARN] Error fetching metadata for {imdb_id}: {e}")
        return None, None


def search_full_movie(title, year=None):
    """
    Search YouTube for the full movie.
    Tries multiple search strategies to find the longest/best match.
    """
    search_queries = [
        f"{title} {year} full movie" if year else f"{title} full movie",
        f"{title} {year} phim trọn bộ" if year else f"{title} phim trọn bộ",
        f"{title} full movie free",
        f"{title} {year} complete movie HD" if year else f"{title} complete movie",
    ]

    for query in search_queries:
        print(f"  Searching: '{query}'")
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(f"ytsearch5:{query}", download=False)

            if not results or "entries" not in results:
                continue

            # Filter for videos >= 60 minutes and sort by duration (longest first)
            candidates = []
            for entry in results["entries"]:
                if entry is None:
                    continue
                duration = entry.get("duration", 0) or 0
                view_count = entry.get("view_count", 0) or 0
                if duration >= MIN_DURATION_SECONDS:
                    candidates.append(
                        {
                            "url": entry.get("webpage_url") or entry.get("url"),
                            "title": entry.get("title", "Unknown"),
                            "duration": duration,
                            "views": view_count,
                        }
                    )

            if candidates:
                # Sort by duration (prefer closest to 90-150 min range) then by views
                candidates.sort(key=lambda x: (-min(x["duration"], 10800), -x["views"]))
                best = candidates[0]
                dur_min = best["duration"] / 60
                print(
                    f"   Found: '{best['title']}' ({dur_min:.0f} min, {best['views']:,} views)"
                )
                return best["url"]

        except Exception as e:
            print(f"  [WARN] Search error: {e}")
            continue

    return None


def download_video(url, output_path):
    """Download a video at 720p max to balance quality and storage."""
    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "outtmpl": str(output_path),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def load_progress():
    """Load download progress to support resuming."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress):
    """Save download progress."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    if not DEPENDENCIES_OK:
        print("Required libraries missing. Please run:")
        print("  conda activate videorag && pip install yt-dlp cinemagoer")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ia = Cinemagoer()
    progress = load_progress()

    print(f"Parsing IMDb IDs from {DVDS_TXT_PATH}...")
    target_movie_ids = get_imdb_ids(DVDS_TXT_PATH, limit=50)

    if not target_movie_ids:
        print("No movie IDs found in dvds.txt.")
        sys.exit(1)

    print(f"Found {len(target_movie_ids)} target movies.")
    print(f"Already completed: {len(progress['completed'])}")
    print(f"Previously failed: {len(progress['failed'])}")
    print(f"Minimum duration filter: {MIN_DURATION_SECONDS // 60} minutes")
    print("=" * 60)

    for i, mid in enumerate(target_movie_ids, 1):
        # Skip already completed
        if mid in progress["completed"]:
            print(
                f"\n[{i}/{len(target_movie_ids)}] {mid} — already downloaded. Skipping."
            )
            continue

        output_file = DOWNLOAD_DIR / f"{mid}.mp4"

        # Check if file already exists and is large enough (> 50 MB = likely a full movie)
        if output_file.exists() and output_file.stat().st_size > 50 * 1024 * 1024:
            print(
                f"\n[{i}/{len(target_movie_ids)}] {mid} — file exists ({output_file.stat().st_size / 1024 / 1024:.0f} MB). Skipping."
            )
            progress["completed"].append(mid)
            save_progress(progress)
            continue

        # Delete small/incomplete files (likely trailers from previous run)
        if output_file.exists():
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(
                f"\n[{i}/{len(target_movie_ids)}] {mid} — existing file too small ({size_mb:.0f} MB), likely a trailer. Re-downloading..."
            )
            output_file.unlink()

        print(f"\n[{i}/{len(target_movie_ids)}] Fetching metadata for {mid}...")
        title, year = get_movie_title(ia, mid)

        if not title:
            print(f"  Could not retrieve title. Skipping.")
            progress["failed"].append(mid)
            save_progress(progress)
            continue

        print(f"  Movie: {title} ({year})")

        # Search for full movie
        video_url = search_full_movie(title, year)

        if not video_url:
            print(f"   No full-length video found for '{title}'. Skipping.")
            progress["failed"].append(mid)
            save_progress(progress)
            continue

        # Download
        print(f"  Downloading full movie to {output_file}...")
        success = download_video(video_url, output_file)

        if success and output_file.exists():
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"   Downloaded {mid} — {title} ({size_mb:.0f} MB)")
            progress["completed"].append(mid)
        else:
            print(f"   Failed to download {mid}")
            progress["failed"].append(mid)

        save_progress(progress)

        # Small delay to be polite to YouTube
        time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE")
    print(f"   Successfully downloaded: {len(progress['completed'])}")
    print(f"   Failed: {len(progress['failed'])}")
    if progress["failed"]:
        print(f"  Failed IDs: {', '.join(progress['failed'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
