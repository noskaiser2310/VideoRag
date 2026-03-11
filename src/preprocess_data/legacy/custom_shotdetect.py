import os
import cv2
import json
from pathlib import Path

RAW_VIDEOS_DIR = Path(r"D:\Study\School\project_ky4\data\raw_videos")
MOVIENET_OUT_DIR = Path(r"D:\Study\School\project_ky4\data\movienet")

SCENE_LENGTH_SEC = 30  # Simulate a scene every 30 seconds


def process_video_to_shots_fast(video_path, imdb_id):
    """
    Simulates Scene Detection by uniformly partitioning the movie into 30s 'shots'.
    Extracts exactly 3 frames per scene like MovieNet to fulfill structural requirements instantly.
    """
    out_img_dir = MOVIENET_OUT_DIR / "shot_keyf" / imdb_id
    out_txt_dir = MOVIENET_OUT_DIR / "shot_txt"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[{imdb_id}] Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # Generate pseudo-scenes
    scene_list = []
    for start_sec in range(0, int(duration_sec), SCENE_LENGTH_SEC):
        end_sec = min(start_sec + SCENE_LENGTH_SEC, int(duration_sec))
        if end_sec > start_sec:
            scene_list.append((int(start_sec * fps), int(end_sec * fps)))

    print(f"[{imdb_id}] Simulated {len(scene_list)} structural scenes (30s each).")

    txt_lines = []

    for shot_idx, scene in enumerate(scene_list):
        start_frame = scene[0]
        end_frame = scene[1]

        # Determine 3 keyframes per shot to emulate MovieNet
        diff = end_frame - start_frame
        if diff < 3:
            k1 = k2 = k3 = start_frame
        else:
            k1 = start_frame + diff // 4
            k2 = start_frame + diff // 2
            k3 = start_frame + 3 * diff // 4

        txt_lines.append(f"{start_frame} {end_frame} {k1} {k2} {k3}")

        # Extract the images efficiently
        for map_i, k_frame in enumerate([k1, k2, k3]):
            out_img_path = out_img_dir / f"shot_{shot_idx:04d}_img_{map_i}.jpg"
            if out_img_path.exists():
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, k_frame)
            ret, frame = cap.read()
            if ret:
                # Resize to standard size (MovieNet is usually ~300-400px tall to save space)
                # Resize width to maintain aspect ratio, height to 360
                h, w = frame.shape[:2]
                new_h = 360
                new_w = int(w * (new_h / h))
                if new_w > 0:
                    frame = cv2.resize(frame, (new_w, new_h))
                cv2.imwrite(str(out_img_path), frame)

    cap.release()

    txt_path = out_txt_dir / f"{imdb_id}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

    print(
        f"[{imdb_id}] Successfully extracted {len(scene_list)} shots into {out_img_dir}"
    )


def main():
    # Only process files that exist to avoid blocking
    downloaded_movies = [
        p.stem
        for p in RAW_VIDEOS_DIR.glob("*.mp4")
        if p.stat().st_size > 50 * 1024 * 1024
    ]
    print(f"Found {len(downloaded_movies)} completely downloaded movies via disk scan.")

    for imdb_id in sorted(downloaded_movies):
        video_path = RAW_VIDEOS_DIR / f"{imdb_id}.mp4"

        # Check if already processed
        out_txt_path = MOVIENET_OUT_DIR / "shot_txt" / f"{imdb_id}.txt"
        if out_txt_path.exists():
            print(f"Skipping {imdb_id}, shot splits already generated.")
            continue

        try:
            process_video_to_shots_fast(video_path, imdb_id)
        except Exception as e:
            print(f"Error processing {imdb_id}: {e}")


if __name__ == "__main__":
    main()
