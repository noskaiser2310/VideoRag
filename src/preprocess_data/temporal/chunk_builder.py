"""
5-Layer Temporal Chunk Builder

Builds temporal chunks by merging 5 data layers:
  ① Timestamps (from annotation or keyframe_index.json)
  ② Semantics (from MovieGraphs clips or auto-generated)
  ③ Dialogue (from SRT subtitles)
  ④ Metadata (from meta JSON)
  ⑤ Keyframes (from shot_keyf/ with precise timestamps)

Supports TWO modes:
  A) Annotated: pre-existing annotation + MovieGraphs data
  B) Ingest: NEW video with no prior data — uses scene detection + STT

Adapted from: scripts/build_temporal_chunks.py
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from ..config import PreprocessConfig as Cfg
from .subtitle_parser import SubtitleParser

logger = logging.getLogger(__name__)


class ChunkBuilder:
    """Build 5-layer temporal chunks for movies."""

    def __init__(self):
        self._keyframe_index_cache: Dict[str, List[Dict]] = {}

    # 
    # Layer ⑤: Keyframe time index (with precision + fallback)
    # 

    def _load_keyframe_index(self, movie_id: str) -> List[Dict]:
        """Load keyframes with timestamps (from index JSON or heuristic)."""
        if movie_id in self._keyframe_index_cache:
            return self._keyframe_index_cache[movie_id]

        # Find keyframe directory
        keyf_dir = None
        for d in Cfg.KEYF_SEARCH_DIRS:
            candidate = d / movie_id
            if candidate.exists() and any(candidate.glob("shot_*.jpg")):
                keyf_dir = candidate
                break

        if not keyf_dir:
            self._keyframe_index_cache[movie_id] = []
            return []

        # Strategy A: keyframe_index.json (precise timestamps)
        idx_path = keyf_dir / "keyframe_index.json"
        if idx_path.exists():
            try:
                data = json.loads(idx_path.read_text(encoding="utf-8"))
                result = [
                    {
                        "path": kf["path"],
                        "timestamp_sec": kf["timestamp_sec"],
                        "scene_idx": kf.get("scene_idx", 0),
                        "img_idx": kf.get("img_idx", 0),
                        "source": "index_json",
                    }
                    for kf in data.get("keyframes", [])
                ]
                self._keyframe_index_cache[movie_id] = result
                return result
            except Exception as e:
                logger.warning(f"Failed to read keyframe index: {e}")

        # Strategy B: Heuristic (shot_num × 3s)
        result = []
        for img_path in sorted(keyf_dir.glob("shot_*_img_*.jpg")):
            match = re.match(r"shot_(\d+)_img_(\d+)\.jpg", img_path.name)
            if match:
                shot_num = int(match.group(1))
                img_idx = int(match.group(2))
                result.append(
                    {
                        "path": str(img_path),
                        "timestamp_sec": shot_num * Cfg.KEYFRAME_INTERVAL_SEC,
                        "scene_idx": shot_num,
                        "img_idx": img_idx,
                        "source": "heuristic_3s",
                    }
                )

        self._keyframe_index_cache[movie_id] = result
        return result

    def _find_keyframes_by_time(
        self, movie_id: str, start_sec: float, end_sec: float
    ) -> List[str]:
        """Find keyframes in [start_sec, end_sec] time range."""
        kf_index = self._load_keyframe_index(movie_id)
        if not kf_index or end_sec <= start_sec:
            return []

        # Group by scene_idx, pick best per scene
        groups: Dict[int, List[Dict]] = {}
        for kf in kf_index:
            if start_sec <= kf["timestamp_sec"] <= end_sec:
                si = kf["scene_idx"]
                groups.setdefault(si, []).append(kf)

        IMG_PRIORITY = {1: 0, 0: 1, 2: 2}
        paths = []
        for si in sorted(groups.keys()):
            best = sorted(groups[si], key=lambda x: IMG_PRIORITY.get(x["img_idx"], 99))
            paths.append(best[0]["path"])
        return paths

    # 
    # Build chunks for a movie
    # 

    def build_for_movie(
        self,
        movie_id: str,
        unified_data: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Build all temporal chunks for a single movie.

        Works in TWO modes:
          A) If annotation + MovieGraphs data exists → clip-centric with scene anchoring
          B) If only keyframe_index.json exists (new video) → scene-centric from keyframe index
        """
        logger.info(f"\n{'=' * 50}\n  Building chunks: {movie_id}\n{'=' * 50}")

        # Load all layers
        scenes = self._load_annotation_scenes(movie_id)
        clips = self._load_clips(movie_id, unified_data or {})
        meta = self._load_meta(movie_id)
        srt_entries = SubtitleParser.load_for_movie(movie_id)

        title = meta.get("title", movie_id)
        genres = meta.get("genres", [])
        cast_map = {
            c.get("name", ""): c.get("character", "") for c in meta.get("cast", [])
        }

        chunks = []

        if scenes and clips:
            #  Mode A: Clip-centric with annotation scene timestamps 
            chunks = self._build_clip_centric(
                movie_id, clips, scenes, srt_entries, title, genres, cast_map
            )
        elif scenes:
            #  Mode A (scene-only): No MovieGraphs but has annotations 
            chunks = self._build_scene_only(
                movie_id, scenes, srt_entries, title, genres
            )
        else:
            #  Mode B: NEW VIDEO — no annotation, use keyframe_index.json 
            kf_index = self._load_keyframe_index(movie_id)
            if kf_index:
                chunks = self._build_from_keyframe_index(
                    movie_id, kf_index, srt_entries, title, genres
                )
            else:
                logger.warning(f"  {movie_id}: No data sources found. Skipping.")

        # Stats
        logger.info(
            f"  [RESULT] {movie_id}: {len(chunks)} chunks, "
            f"{sum(1 for c in chunks if c['keyframe_paths'])} with keyframes, "
            f"{sum(1 for c in chunks if c['dialogue'])} with dialogue"
        )
        return chunks

    #  Mode A: Clip-centric (annotated dataset) 

    def _build_clip_centric(
        self, movie_id, clips, scenes, srt, title, genres, cast_map
    ):
        chunks = []
        for i, clip in enumerate(clips):
            matched = self._match_clip_to_scene(clip, scenes)
            if matched:
                start_sec, end_sec = matched["start_seconds"], matched["end_seconds"]
                ts_source = "annotation_frame"
            else:
                start_sec, end_sec = 0, 0
                ts_source = "none"

            dialogues = (
                SubtitleParser.align(srt, start_sec, end_sec)
                if start_sec or end_sec
                else []
            )
            keyframes = self._find_keyframes_by_time(movie_id, start_sec, end_sec)

            chunks.append(
                self._make_chunk(
                    movie_id,
                    i,
                    title,
                    genres,
                    start_sec,
                    end_sec,
                    ts_source,
                    clip=clip,
                    dialogues=dialogues,
                    keyframes=keyframes,
                    cast_map=cast_map,
                )
            )
        return chunks

    #  Mode A (scene-only) 

    def _build_scene_only(self, movie_id, scenes, srt, title, genres):
        chunks = []
        for i, scene in enumerate(scenes):
            s, e = scene["start_seconds"], scene["end_seconds"]
            dialogues = SubtitleParser.align(srt, s, e)
            keyframes = self._find_keyframes_by_time(movie_id, s, e)

            chunks.append(
                self._make_chunk(
                    movie_id,
                    i,
                    title,
                    genres,
                    s,
                    e,
                    "annotation_frame",
                    dialogues=dialogues,
                    keyframes=keyframes,
                    scene_label=scene.get("place_tag", ""),
                )
            )
        return chunks

    #  Mode B: NEW VIDEO — from keyframe_index.json scenes 

    def _build_from_keyframe_index(self, movie_id, kf_index, srt, title, genres):
        """Build chunks from keyframe_index.json for new videos with no annotation."""
        # Group keyframes by scene_idx
        scene_groups: Dict[int, List[Dict]] = {}
        for kf in kf_index:
            si = kf["scene_idx"]
            scene_groups.setdefault(si, []).append(kf)

        chunks = []
        for i, si in enumerate(sorted(scene_groups.keys())):
            kfs = scene_groups[si]
            start_sec = min(kf["timestamp_sec"] for kf in kfs)
            end_sec = max(kf["timestamp_sec"] for kf in kfs)
            # Expand range for single-keyframe scenes
            if end_sec == start_sec:
                end_sec = start_sec + Cfg.KEYFRAME_INTERVAL_SEC

            dialogues = SubtitleParser.align(srt, start_sec, end_sec)
            paths = [kf["path"] for kf in sorted(kfs, key=lambda x: x["img_idx"])]

            chunks.append(
                self._make_chunk(
                    movie_id,
                    i,
                    title,
                    genres,
                    start_sec,
                    end_sec,
                    "keyframe_index",
                    dialogues=dialogues,
                    keyframes=paths,
                )
            )
        return chunks

    #  Chunk factory 

    @staticmethod
    def _make_chunk(
        movie_id,
        idx,
        title,
        genres,
        start_sec,
        end_sec,
        ts_source,
        clip=None,
        dialogues=None,
        keyframes=None,
        cast_map=None,
        scene_label="",
    ) -> Dict:
        dialogues = dialogues or []
        keyframes = keyframes or []
        clip = clip or {}
        cast_map = cast_map or {}

        def _fmt(s):
            h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        characters = clip.get("characters", [])

        return {
            "chunk_id": f"{movie_id}_chunk_{idx:04d}",
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            # Temporal
            "start_time": _fmt(start_sec),
            "end_time": _fmt(end_sec),
            "start_seconds": round(start_sec, 2),
            "end_seconds": round(end_sec, 2),
            "duration_seconds": round(end_sec - start_sec, 2),
            "timestamp_source": ts_source,
            # Semantic
            "clip_id": clip.get("clip_id", ""),
            "description": clip.get("description", ""),
            "situation": clip.get("situation", ""),
            "scene_label": clip.get("scene_label", scene_label),
            "characters": characters,
            "character_ids": clip.get("character_ids", []),
            "attributes": clip.get("attributes", []),
            "interactions": clip.get("interactions", []),
            # Dialogue
            "dialogue": dialogues,
            "dialogue_text": " ".join(dialogues) if dialogues else "",
            # Shot range
            "shot_start": clip.get("start_shot", 0),
            "shot_end": clip.get("end_shot", 0),
            # Keyframes
            "keyframe_paths": keyframes,
            "num_keyframes": len(keyframes),
            # Cast mapping
            "cast_in_scene": [
                {"actor": actor, "character": char}
                for char_name in characters
                for actor, char in cast_map.items()
                if char_name.lower() in char.lower()
                or char.lower() in char_name.lower()
            ],
        }

    #  Helpers 

    @staticmethod
    def _load_annotation_scenes(movie_id: str) -> List[Dict]:
        ann_path = Cfg.ANNOTATION_DIR / f"{movie_id}.json"
        if not ann_path.exists():
            return []
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        # Get FPS from keyframe_index or default
        fps = 24.0
        for d in Cfg.KEYF_SEARCH_DIRS:
            idx_path = d / movie_id / "keyframe_index.json"
            if idx_path.exists():
                try:
                    idx = json.loads(idx_path.read_text(encoding="utf-8"))
                    fps = idx.get("video_fps", 24.0)
                except Exception:
                    pass
                break

        scenes = []
        for i, s in enumerate(data.get("scene", [])):
            fr = s.get("frame", [0, 0])
            sh = s.get("shot", [0, 0])
            if len(fr) < 2 or len(sh) < 2:
                continue
            ss, es = fr[0] / fps, fr[1] / fps
            scenes.append(
                {
                    "scene_idx": i,
                    "shot_start": sh[0],
                    "shot_end": sh[1],
                    "start_seconds": round(ss, 2),
                    "end_seconds": round(es, 2),
                    "duration_seconds": round(es - ss, 2),
                    "place_tag": s.get("place_tag"),
                    "action_tag": s.get("action_tag"),
                }
            )
        return scenes

    @staticmethod
    def _load_clips(movie_id: str, unified_data: Dict) -> List[Dict]:
        movie = unified_data.get("movies", {}).get(movie_id)
        if not movie:
            return []
        return [
            {
                "clip_id": c.get("clip_id", ""),
                "start_shot": c.get("start_shot", 0),
                "end_shot": c.get("end_shot", 0),
                "description": c.get("description", ""),
                "situation": c.get("situation", ""),
                "scene_label": c.get("scene_label", ""),
                "characters": [ch.get("name", "") for ch in c.get("characters", [])],
                "character_ids": [ch.get("id", "") for ch in c.get("characters", [])],
                "attributes": list(set(c.get("attributes", []))),
                "interactions": c.get("interactions", []),
            }
            for c in movie.get("clips", [])
        ]

    @staticmethod
    def _load_meta(movie_id: str) -> Dict:
        for d in Cfg.META_SEARCH_DIRS:
            p = d / f"{movie_id}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {}

    @staticmethod
    def _match_clip_to_scene(clip, scenes):
        clip_start, clip_end = clip["start_shot"], clip["end_shot"]
        best, best_overlap = None, 0
        for scene in scenes:
            overlap = max(
                0,
                min(clip_end, scene["shot_end"]) - max(clip_start, scene["shot_start"]),
            )
            if overlap > best_overlap:
                best_overlap, best = overlap, scene
        if not best:
            for scene in scenes:
                if scene["shot_start"] <= clip_start < scene["shot_end"]:
                    return scene
        return best

    # 
    # Batch build + save
    # 

    def build_all(self, movie_ids: List[str] = None) -> Dict:
        """Build chunks for all movies and save to disk."""
        # Load unified dataset
        unified_data = {}
        if Cfg.UNIFIED_DATASET_JSON.exists():
            unified_data = json.loads(
                Cfg.UNIFIED_DATASET_JSON.read_text(encoding="utf-8")
            )

        if movie_ids is None:
            movie_ids = Cfg.get_all_movie_ids()

        Cfg.TEMPORAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        all_chunks = []
        stats = {"movies": 0, "chunks": 0}

        for mid in movie_ids:
            chunks = self.build_for_movie(mid, unified_data)
            if chunks:
                # Save per-movie
                out = Cfg.TEMPORAL_CHUNKS_DIR / f"{mid}_chunks.json"
                out.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                all_chunks.extend(chunks)
                stats["movies"] += 1
                stats["chunks"] += len(chunks)

        # Save merged
        merged = {
            "metadata": {
                "total_movies": stats["movies"],
                "total_chunks": stats["chunks"],
            },
            "chunks": all_chunks,
        }
        merged_path = Cfg.TEMPORAL_CHUNKS_DIR / "all_chunks.json"
        merged_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            f"\n   Built {stats['chunks']} chunks for {stats['movies']} movies → {merged_path}"
        )
        return merged
