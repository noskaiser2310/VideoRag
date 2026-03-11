import os
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(r"D:\Study\School\project_ky4\data\movienet_tools")))
from movienet.tools import ShotDetector

RAW_VIDEOS_DIR = Path(r"D:\Study\School\project_ky4\data\raw_videos")
MOVIENET_OUT_DIR = Path(r"D:\Study\School\project_ky4\data\movienet")


def main():
    sdt = ShotDetector(
        print_list=True,
        save_keyf=True,  # Save the 3 keyframes per shot
        save_keyf_txt=True,  # Save the timestamps
        split_video=False,  # Don't split the actual video to save space
    )

    # Process the ones that are successfully downloaded
    progress_file = RAW_VIDEOS_DIR / "_goojara_progress.json"
    with open(progress_file, "r") as f:
        data = json.load(f)

    downloaded_movies = data.get("downloaded", [])
    if not downloaded_movies:
        # Fallback to scanning raw_videos
        downloaded_movies = [p.stem for p in RAW_VIDEOS_DIR.glob("*.mp4")]

    print(f"Found {len(downloaded_movies)} downloaded movies.")

    for imdb_id in downloaded_movies:
        video_path = RAW_VIDEOS_DIR / f"{imdb_id}.mp4"
        if not video_path.exists():
            continue

        out_shot_dir = MOVIENET_OUT_DIR / "shot_keyf" / imdb_id
        if out_shot_dir.exists() and len(list(out_shot_dir.glob("*.jpg"))) > 0:
            print(f"Skipping {imdb_id}, already processed.")
            continue

        print(f"Processing {imdb_id}...")
        try:
            # ShotDetector automatically creates the 'shot_keyf', 'shot_txt', 'shot_stats' subfolders inside out_dir
            sdt.shotdetect(str(video_path), str(MOVIENET_OUT_DIR))
        except Exception as e:
            print(f"Error processing {imdb_id}: {e}")


if __name__ == "__main__":
    main()
