import os
import shutil
from pathlib import Path

TARGET_MOVIES = [
    "tt0097576",
    "tt0109830",
    "tt0110912",
    "tt0241527",
    "tt0375679",
    "tt1189340",
    "tt1193138",
    "tt1907668",
    "tt0100405",
    "tt0106918",
    "tt0108160",
    "tt0118715",
    "tt0118842",
    "tt0120338",
    "tt0147800",
    "tt0167404",
    "tt0212338",
    "tt0286106",
    "tt0116695",
    "tt0467406",
]


def clean_directory(dir_path: Path, is_file_based=False):
    if not dir_path.exists():
        return

    print(f"Cleaning {dir_path}...")
    removed_count = 0

    for item in dir_path.iterdir():
        imdb_id = None

        # Try to extract IMDB ID
        if item.is_dir() and item.name.startswith("tt"):
            imdb_id = item.name
        elif item.is_file() and item.name.startswith("tt"):
            imdb_id = item.name.split(".")[0].split("_")[
                0
            ]  # handle tt123.mp4 or tt123_abc.json

        if imdb_id and imdb_id.startswith("tt"):
            if imdb_id not in TARGET_MOVIES:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                        print(f"  Removed directory: {item.name}")
                    else:
                        item.unlink()
                        print(f"  Removed file: {item.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"  Failed to remove {item.name}: {e}")

    print(f"Removed {removed_count} items from {dir_path}")


def main():
    # 1. Raw videos
    raw_vids = Path(r"D:\Study\School\project_ky4\data\raw_videos")
    clean_directory(raw_vids)

    # 2. MovieNet extracted keyframes
    movienet_dir = Path(r"D:\Study\School\project_ky4\data\movienet\shot_keyf")
    clean_directory(movienet_dir)

    # 3. Old movie_data_subset (Complete removal requested earlier but failed)
    old_subset = Path(r"D:\Study\School\project_ky4\movie_data_subset")
    if old_subset.exists():
        print(f"Removing entirely: {old_subset}")
        try:
            shutil.rmtree(old_subset)
            print("Successfully removed old movie_data_subset")
        except Exception as e:
            print(f"Failed to remove {old_subset}: {e}")

    # 4. We can also clean the `movie_data/files` directly, though they might be needed for backup.
    # For now, let's clean the main bulk data locations.

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
