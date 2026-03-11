"""
Unified Dataset Builder for MovieRAG.
Merges MovieNet keyframes + MovieGraphs scene descriptions into a single JSON dataset.
Output: data/unified_dataset/movierag_dataset.json
"""

import os
import sys
import json
import pickle
import glob
from pathlib import Path
from collections import OrderedDict

# Paths
MOVIEGRAPHS_PKL = Path(
    r"D:\Study\School\project_ky4\data\MovieGraphs_repo\py3loader_new\all_movies.pkl"
)
MOVIEGRAPHS_LOADER = Path(
    r"D:\Study\School\project_ky4\data\MovieGraphs_repo\py3loader_new"
)
MOVIENET_KEYF_DIR = Path(r"D:\Study\School\project_ky4\data\movienet\shot_keyf")
DVDS_TXT = Path(r"D:\Study\School\project_ky4\data\MovieGraphs_repo\dvds.txt")
OUTPUT_DIR = Path(r"D:\Study\School\project_ky4\movie_data_subset_20\unified_dataset")
OUTPUT_JSON = OUTPUT_DIR / "movierag_dataset.json"

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


def get_target_movies(dvds_path):
    """Extract IMDb IDs and titles from dvds.txt but only keep TARGET_MOVIES."""
    movies = []
    with open(dvds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 3 and parts[1].startswith("tt"):
                    imdb_id = parts[1]
                    if imdb_id not in TARGET_MOVIES:
                        continue
                    # Title is everything after the IMDb ID, minus the year at the end
                    rest = " ".join(parts[2:])
                    # Year is last 4-digit number
                    year = (
                        parts[-1]
                        if parts[-1].isdigit() and len(parts[-1]) == 4
                        else None
                    )
                    title = " ".join(parts[2:-1]) if year else rest
                    movies.append({"imdb_id": imdb_id, "title": title, "year": year})
    return movies


def load_moviegraphs():
    """Load MovieGraphs all_movies.pkl."""
    sys.path.insert(0, str(MOVIEGRAPHS_LOADER))
    with open(MOVIEGRAPHS_PKL, "rb") as f:
        data = pickle.load(f)
    return data


def extract_clip_info(clip_graph):
    """Extract structured info from a ClipGraph object."""
    info = {
        "situation": getattr(clip_graph, "situation", ""),
        "scene_label": getattr(clip_graph, "scene_label", ""),
        "description": getattr(clip_graph, "description", ""),
    }

    # Video/shot info
    video = getattr(clip_graph, "video", {})
    if video:
        info["start_shot"] = video.get("ss", None)
        info["end_shot"] = video.get("es", None)
        info["scene_ids"] = video.get("scene", [])

    # Extract characters
    try:
        chars = clip_graph.get_characters()
        info["characters"] = (
            [{"name": c[0], "id": c[1]} for c in chars] if chars else []
        )
    except Exception:
        info["characters"] = []

    # Extract entities, attributes
    try:
        node_dict = clip_graph.get_node_type_dict()
        info["entities"] = node_dict.get("entity", [])
        info["attributes"] = node_dict.get("attribute", [])
        info["interactions"] = node_dict.get("interaction", [])
        info["relationships"] = node_dict.get("relationship", [])
    except Exception:
        pass

    return info


def get_keyframes_for_movie(imdb_id):
    """Get all keyframe image paths for a movie from MovieNet."""
    movie_dir = MOVIENET_KEYF_DIR / imdb_id
    if not movie_dir.exists():
        return []

    keyframes = []
    for img_path in sorted(movie_dir.glob("*.jpg")):
        fname = img_path.name
        # Parse shot info from filename: shot_XXXX_img_Y.jpg
        parts = fname.replace(".jpg", "").split("_")
        shot_id = None
        img_idx = None
        for i, p in enumerate(parts):
            if p == "shot" and i + 1 < len(parts):
                try:
                    shot_id = int(parts[i + 1])
                except ValueError:
                    pass
            if p == "img" and i + 1 < len(parts):
                try:
                    img_idx = int(parts[i + 1])
                except ValueError:
                    pass

        keyframes.append(
            {
                "filename": fname,
                "path": str(img_path),
                "shot_id": shot_id,
                "img_index": img_idx,
            }
        )
    return keyframes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get target movie list
    print("Step 1: Reading target movie list from dvds.txt...")
    target_movies = get_target_movies(DVDS_TXT)
    print(f"  Found {len(target_movies)} target movies.")

    # 2. Load MovieGraphs
    print("Step 2: Loading MovieGraphs data (all_movies.pkl)...")
    mg_data = load_moviegraphs()
    mg_keys = list(mg_data.keys())
    print(f"  Loaded {len(mg_keys)} movies from MovieGraphs.")

    # 3. Build unified dataset
    print("Step 3: Building unified dataset...")
    dataset = {
        "metadata": {
            "name": "MovieRAG Unified Dataset",
            "version": "1.0",
            "description": "Merged dataset from MovieNet (keyframes) and MovieGraphs (scene descriptions)",
            "num_movies": 0,
            "total_clips": 0,
            "total_keyframes": 0,
        },
        "movies": {},
    }

    total_clips = 0
    total_keyframes = 0

    for movie_info in target_movies:
        imdb_id = movie_info["imdb_id"]
        print(f"\n  Processing {imdb_id} ({movie_info['title']})...")

        movie_entry = {
            "imdb_id": imdb_id,
            "title": movie_info["title"],
            "year": movie_info["year"],
            "sources": {"movienet": False, "moviegraphs": False},
            "keyframes": [],
            "clips": [],
        }

        # MovieNet keyframes
        keyframes = get_keyframes_for_movie(imdb_id)
        if keyframes:
            movie_entry["keyframes"] = keyframes
            movie_entry["sources"]["movienet"] = True
            total_keyframes += len(keyframes)
            print(f"    MovieNet: {len(keyframes)} keyframes")
        else:
            print(f"    MovieNet: No keyframes found")

        # MovieGraphs clips
        mg_movie = mg_data.get(imdb_id)
        if mg_movie is not None:
            movie_entry["sources"]["moviegraphs"] = True

            # Extract castlist
            castlist = getattr(mg_movie, "castlist", None)
            if castlist:
                movie_entry["cast"] = (
                    [
                        {"name": c.get("name", ""), "id": c.get("chid", "")}
                        for c in castlist
                    ]
                    if isinstance(castlist, list)
                    else []
                )

            # Extract clips/scenes
            clip_graphs = getattr(mg_movie, "clip_graphs", {})
            if not clip_graphs:
                # Try alternative attribute names
                for attr in ["clipgraphs", "clips", "scenes"]:
                    clip_graphs = getattr(mg_movie, attr, {})
                    if clip_graphs:
                        break

            if clip_graphs:
                for sid, cg in clip_graphs.items():
                    clip_info = extract_clip_info(cg)
                    clip_info["clip_id"] = str(sid)
                    movie_entry["clips"].append(clip_info)
                total_clips += len(movie_entry["clips"])
                print(f"    MovieGraphs: {len(movie_entry['clips'])} scene clips")
            else:
                # Try to get any attribute that looks like it contains clips
                mg_attrs = [a for a in dir(mg_movie) if not a.startswith("_")]
                print(
                    f"    MovieGraphs: Found movie object, attributes: {mg_attrs[:10]}"
                )
        else:
            print(f"    MovieGraphs: Not found in pkl data")

        dataset["movies"][imdb_id] = movie_entry

    # Update metadata
    dataset["metadata"]["num_movies"] = len(dataset["movies"])
    dataset["metadata"]["total_clips"] = total_clips
    dataset["metadata"]["total_keyframes"] = total_keyframes

    # 4. Save
    print(f"\nStep 4: Saving unified dataset to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)

    file_size = OUTPUT_JSON.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f" UNIFIED DATASET CREATED SUCCESSFULLY")
    print(f"  Movies: {dataset['metadata']['num_movies']}")
    print(f"  Total Scene Clips (MovieGraphs): {total_clips}")
    print(f"  Total Keyframes (MovieNet): {total_keyframes}")
    print(f"  File Size: {file_size:.1f} MB")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
