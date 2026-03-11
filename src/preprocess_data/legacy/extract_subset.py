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

print(f"Total target IDs to process: {len(TARGET_MOVIES)}")

# Paths
BASE_FILES_DIR = Path(r"D:\Study\School\project_ky4\movie_data\files")
SUBSET_DIR = Path(r"D:\Study\School\project_ky4\movie_data_subset_20")

folders = ["annotation", "meta", "script", "subtitle"]

for folder in folders:
    src_folder = BASE_FILES_DIR / folder
    dst_folder = SUBSET_DIR / folder

    if not src_folder.exists():
        print(f"Source folder does not exist: {src_folder}")
        continue

    # Create destination folder
    os.makedirs(dst_folder, exist_ok=True)

    # Copy files matching the target IDs
    copied = 0
    # iterdir() does not guarantee sorted order, but it's fine for copying
    for file_path in src_folder.iterdir():
        if not file_path.is_file():
            continue

        file_name = file_path.name
        # Check if file name starts with any of the target IDs
        for target_id in TARGET_MOVIES:
            if file_name.startswith(target_id):
                shutil.copy2(file_path, dst_folder / file_name)
                copied += 1
                break

    print(f"Copied {copied} files to {dst_folder}")

print("Extraction complete.")
