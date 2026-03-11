import json, os

os.chdir(r"D:\Study\School\project_ky4")
d = json.load(
    open("data/movienet/shot_keyf/tt0120338/keyframe_index.json", "r", encoding="utf-8")
)
print(f"Total keyframes: {d['total_keyframes']}")
print(f"Total scenes: {d['total_scenes']}")
print(f"FPS: {d['video_fps']}")
kf = d["keyframes"][0]
print(f"First: {kf['filename']} at {kf['timestamp_fmt']} ({kf['timestamp_sec']}s)")
kf = d["keyframes"][-1]
print(f"Last: {kf['filename']} at {kf['timestamp_fmt']} ({kf['timestamp_sec']}s)")
kf = d["keyframes"][10]
print(
    f"Sample: {kf['filename']} at {kf['timestamp_fmt']} ({kf['timestamp_sec']}s) scene_{kf['scene_idx']}"
)
