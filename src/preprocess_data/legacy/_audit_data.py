import json, os
from pathlib import Path

os.chdir(r"D:\Study\School\project_ky4")

# 1. Temporal chunks stats
d = json.load(open("data/temporal_chunks/all_chunks.json", "r", encoding="utf-8"))
chunks = d["chunks"]
movies = set(c["movie_id"] for c in chunks)
kf = sum(c["num_keyframes"] for c in chunks)
dial = sum(1 for c in chunks if c["dialogue"])
desc = sum(1 for c in chunks if c["description"])
ts_sources = {}
for c in chunks:
    s = c.get("timestamp_source", "?")
    ts_sources[s] = ts_sources.get(s, 0) + 1

print("=== TEMPORAL CHUNKS ===")
print(f"Total chunks: {len(chunks)}")
print(f"Movies: {len(movies)} → {sorted(movies)}")
print(f"Total keyframes: {kf}")
print(f"With dialogue: {dial}")
print(f"With description: {desc}")
print(f"Timestamp sources: {ts_sources}")

# 2. Index sizes
print("\n=== INDEXES ===")
for p in sorted(Path("data/indexes").glob("*")):
    size = p.stat().st_size / (1024 * 1024)
    print(f"  {p.name}: {size:.1f} MB")

# 3. Unified dataset
print("\n=== UNIFIED DATASET ===")
for p in sorted(Path("data/unified_dataset").glob("*")):
    size = p.stat().st_size / (1024 * 1024)
    print(f"  {p.name}: {size:.1f} MB")

# 4. Annotation count
ann_dir = Path("data/movienet_subset/annotation")
if ann_dir.exists():
    print(f"\n=== ANNOTATIONS === {len(list(ann_dir.glob('*.json')))} files")

# 5. Subtitle count
sub_dir = Path("data/movienet_subset/subtitle")
if sub_dir.exists():
    print(f"=== SUBTITLES === {len(list(sub_dir.glob('*.srt')))} files")

# 6. Meta count
meta_dir = Path("data/movienet_subset/meta")
if meta_dir.exists():
    print(f"=== META === {len(list(meta_dir.glob('*.json')))} files")

# 7. Script count
script_dir = Path("data/movienet_subset/script")
if script_dir.exists():
    print(f"=== SCRIPTS === {len(list(script_dir.glob('*')))} files")

# 8. Keyframe dirs
kf_dir = Path("data/movienet/shot_keyf")
if kf_dir.exists():
    for mid in sorted(p.name for p in kf_dir.iterdir() if p.is_dir()):
        imgs = len(list((kf_dir / mid).glob("*.jpg")))
        has_idx = (kf_dir / mid / "keyframe_index.json").exists()
        print(f"  {mid}: {imgs} images, index={'' if has_idx else ''}")

# 9. Raw videos
vid_dir = Path("data/raw_videos")
if vid_dir.exists():
    for p in sorted(vid_dir.glob("*.*")):
        size = p.stat().st_size / (1024 * 1024 * 1024)
        print(f"  {p.name}: {size:.1f} GB")

# 10. Dataset sample
ds = json.load(
    open("data/unified_dataset/movierag_dataset.json", "r", encoding="utf-8")
)
print(
    f"\n=== DATASET === {ds['metadata']['num_movies']} movies, {ds['metadata']['total_clips']} clips, {ds['metadata']['total_keyframes']} keyframes"
)
sample_mid = list(ds["movies"].keys())[0]
m = ds["movies"][sample_mid]
print(f"  Sample: {sample_mid} - {m.get('title', '?')}")
print(f"    Sources: {m.get('sources', {})}")
print(f"    Clips: {len(m.get('clips', []))}")
print(f"    Keyframes: {len(m.get('keyframes', []))}")
if m.get("clips"):
    c = m["clips"][0]
    print(f"    Sample clip: desc={c.get('description', '')[:80]}")
    print(
        f"      shots: {c.get('start_shot')}→{c.get('end_shot')}, chars: {[ch.get('name') for ch in c.get('characters', [])]}"
    )
