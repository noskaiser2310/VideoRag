import cv2
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path(r"D:\Study\School\project_ky4\data")
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
MOVIENET_DIR = DATA_DIR / "movienet" / "shot_keyf"
UNIFIED_DATASET = DATA_DIR / "unified_dataset" / "movierag_dataset.json"


def build_hybrid_extraction_plan():
    """Determine extraction paths for all 51 movies (all unified under shot_keyf now)."""
    if not UNIFIED_DATASET.exists():
        logger.error(f"Dataset not found at {UNIFIED_DATASET}")
        return None

    with open(UNIFIED_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    plan = {"raw_videos": [], "movienet_fallbacks": [], "movies_processed": 0}

    for imdb_id, movie_info in dataset["movies"].items():
        # Everything has formalized into shot_keyf folders now.
        # Check if the movienet shot_keyf folder has images, otherwise no fallback.
        shot_dir = MOVIENET_DIR / imdb_id
        if shot_dir.exists() and len(list(shot_dir.glob("*.jpg"))) > 0:
            plan["movienet_fallbacks"].append(
                {
                    "imdb_id": imdb_id,
                    "path": str(shot_dir),
                    "title": movie_info["title"],
                }
            )
        else:
            logger.warning(f"No keyframes exist for {imdb_id} at {shot_dir}")

        plan["movies_processed"] += 1

    return plan


def run_hybrid_indexing(plan):
    """Pass the plan into the ParallelVisualIndexer."""
    if not plan:
        return

    logger.info("Executing Hybrid Extraction and Indexing...")
    from movierag.indexing.parallel_indexer import ParallelVisualIndexer

    # We will build a single unified Visual Index for all 51 movies
    idx_dir = str(DATA_DIR / "unified_dataset")
    indexer = ParallelVisualIndexer(index_dir=idx_dir, index_name="movie_hybrid_index")

    tasks = []

    logger.info(
        f"Adding {len(plan['movienet_fallbacks'])} MovieNet-structured folders to index tasks (Max 20 shots per folder)..."
    )
    for m in plan["movienet_fallbacks"]:
        folder = Path(m["path"])
        imdb_id = m["imdb_id"]
        if folder.exists() and folder.is_dir():
            all_imgs = sorted(folder.glob("*.jpg"))

            # Group by shot
            from collections import defaultdict

            shots = defaultdict(list)
            for img_path in all_imgs:
                # fname format: shot_0001_img_0.jpg
                parts = img_path.stem.split("_")
                shot_name = "_".join(parts[:2]) if len(parts) >= 2 else "unknown"
                shots[shot_name].append(img_path)

            # Randomly sample up to 20 shots to keep index sizes manageable
            import random

            shot_keys = list(shots.keys())
            sampled_shot_keys = (
                random.sample(shot_keys, min(20, len(shot_keys))) if shot_keys else []
            )

            for s_key in sampled_shot_keys:
                for img_path in shots[s_key]:
                    tasks.append(
                        {
                            "keyframe_id": img_path.stem,
                            "keyframe_path": str(img_path),
                            "timestamp_sec": 0,
                            "movie_id": imdb_id,
                            "source": "movienet",
                            "shot_id": s_key,  # explicit tracking for L1 aggregation
                        }
                    )

    logger.info(f"Total Combined extraction/encoding tasks: {len(tasks)}")

    # We execute this in batches to test integration. Limit to a test size if huge.
    # tasks = tasks[:500]

    # Execute Parallel Map Reduce
    indexer.build_index_parallel(
        items=tasks,
        id_key="keyframe_id",
        path_key="keyframe_path",
        movie_id_key="movie_id",
        batch_size=64,
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(r"d:\Study\School\project_ky4\src")))

    plan = build_hybrid_extraction_plan()
    if plan:
        print(f"Total Movies: {plan['movies_processed']}")
        print(f"Raw Videos to Extract: {len(plan['raw_videos'])}")
        print(f"MovieNet Fallbacks: {len(plan['movienet_fallbacks'])}")

        # Save extraction plan
        plan_out = DATA_DIR / "hybrid_extraction_plan.json"
        with open(plan_out, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"Extraction plan saved to {plan_out}")

        run_hybrid_indexing(plan)
