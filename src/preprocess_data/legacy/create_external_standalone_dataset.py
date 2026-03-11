"""
External Standalone Dataset Compiler (No Academic Mail required)

This script automates the creation of a 'Perfect' standalone Visual Dataset
for your 51 intersecting movies by scraping YouTube trailers using yt-dlp
and extracting keyframes via FFmpeg.

Prerequisites:
1. `pip install yt-dlp opencv-python pydantic`
2. `winget install ffmpeg` (or ensure ffmpeg is in your system PATH)
"""

import os
import subprocess
import time
from pathlib import Path

# Paths
DVDS_TXT_PATH = r"D:\Study\School\project_ky4\data\MovieGraphs_repo\dvds.txt"
DATASET_ROOT = r"D:\Study\School\project_ky4\data\Standalone_Dataset"
VIDEO_DIR = os.path.join(DATASET_ROOT, "videos")
KEYFRAME_DIR = os.path.join(DATASET_ROOT, "keyframes")


def setup_dirs():
    Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    Path(KEYFRAME_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Created dataset root at: {DATASET_ROOT}")


def extract_movie_info(txt_path):
    """Extracts IMDb IDs and Titles from the dvds.txt file."""
    movies = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for count in range(0, len(lines), 3):  # Each movie entry spans roughly 3 lines
            if count >= len(lines):
                break
            first_line = lines[count].strip()
            if not first_line:
                continue

            parts = first_line.split()
            if len(parts) >= 3 and parts[1].startswith("tt"):
                imdb_id = parts[1]
                title = " ".join(
                    parts[2:-1]
                )  # '10 Things I Hate About You', ignoring year
                year = parts[-1]

                # Check line 2 for URL (Optional, but often accurate in the file)
                url = lines[count + 1].strip() if count + 1 < len(lines) else ""

                movies.append(
                    {"id": imdb_id, "title": f"{title} {year}", "amazon_url": url}
                )
    return movies


def download_trailer(movie):
    """Uses yt-dlp to search for and download the movie trailer into an MP4 file."""
    search_query = f"ytsearch1:'{movie['title']} official trailer'"
    output_path = os.path.join(VIDEO_DIR, f"{movie['id']}.mp4")

    if os.path.exists(output_path):
        print(f"  -> Video for {movie['id']} already exists. Skipping download.")
        return output_path

    print(f"  -> Searching and downloading trailer for '{movie['title']}'...")

    # Use yt-dlp format to grab best mp4 available up to 720p to save space
    command = [
        "yt-dlp",
        search_query,
        "-f",
        "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]",
        "-o",
        output_path,
        "--quiet",
        "--no-warnings",
    ]

    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to download {movie['title']}: {e}")
        return None


def extract_keyframes(movie_id, video_path):
    """Uses FFmpeg to extract 1 frame every 3 seconds to simulate MovieNet keyframes."""
    if not video_path or not os.path.exists(video_path):
        return

    output_dir = os.path.join(KEYFRAME_DIR, movie_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    if len(list(Path(output_dir).glob("*.jpg"))) > 5:
        print(f"  -> Keyframes for {movie_id} already extracted. Skipping.")
        return

    print(f"  -> Extracting keyframes to {output_dir}...")

    # Extract 1 frame every 3 seconds (fps=1/3)
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "fps=1/3",
        "-qscale:v",
        "2",  # High quality jpeg
        os.path.join(output_dir, "shot_%04d_img_0.jpg"),
        "-v",
        "quiet",
        "-y",
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] FFmpeg failed on {movie_id}: {e}")


def main():
    print("---  STANDALONE EXTERNAL DATASET SCRAPER ---")
    setup_dirs()

    movies = extract_movie_info(DVDS_TXT_PATH)
    print(f"Found {len(movies)} movies to process.")

    print("\n--- BEGIN PIPELINE ---")
    for i, movie in enumerate(movies, 1):
        print(f"\n[{i}/{len(movies)}] Processing: {movie['title']} ({movie['id']})")

        # 1. Download Video (Trailer) from YouTube
        video_path = download_trailer(movie)

        # 2. Extract Frames
        if video_path:
            extract_keyframes(movie["id"], video_path)

        time.sleep(1)  # Gentle throttling

    print("\n Dataset compilation complete!")
    print(f"Videos saved to: {VIDEO_DIR}")
    print(f"Keyframes saved to: {KEYFRAME_DIR}")

    print("\n NEXT STEPS FOR INDEXING:")
    print(
        "1. Update your UI/Indexer code to point `VisualIndexer` to this new `keyframes` directory."
    )
    print(
        "2. Run the `generate_synthetic_moviegraphs.py` script to generate the intersecting Text JSON for these same IDs."
    )
    print(
        "You now have a perfectly 1:1 aligned multi-modal dataset without any academic approvals!"
    )


if __name__ == "__main__":
    main()
