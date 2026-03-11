"""
Scene Analyzer — VLM + LLM → MovieGraphs-equivalent JSON

Analyzes extracted keyframes using:
  1. VLM (Llama 4 Scout via Groq) → visual description per scene
  2. LLM (Kimi K2) → structured JSON: situation, characters, interactions

Generates scene graph data equivalent to MovieGraphs ClipGraph format
for new videos that don't have pre-existing MovieGraphs data.
"""

import json
import logging
import base64
from pathlib import Path
from typing import List, Dict, Optional

from ..config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)

SCENE_ANALYSIS_PROMPT = """Analyze this movie scene keyframe image. Provide a detailed JSON description:

{
  "situation": "brief situation label (e.g. 'car chase', 'romantic dinner')",
  "description": "2-3 sentence description of what's happening in this scene",
  "characters": [
    {"name": "Character Name or 'Unknown Person 1'", "appearance": "brief visual description"}
  ],
  "attributes": ["emotion/mood descriptors like 'tense', 'romantic', 'dark'"],
  "interactions": ["action verbs: 'talking', 'fighting', 'running'"],
  "setting": "location/environment description",
  "objects": ["notable objects visible in scene"]
}

Return ONLY the JSON, no other text."""

STRUCTURE_PROMPT = """You are structuring movie scene analysis data. Given these VLM analyses of keyframes from the same scene, merge them into a single MovieGraphs-compatible clip entry.

VLM Analyses:
{analyses}

Movie metadata:
- Title: {title}
- Movie ID: {movie_id}
- Scene index: {scene_idx}
- Time range: {start_time} → {end_time}

Output a single merged JSON:
{{
  "situation": "consolidated situation label",
  "description": "merged description covering the full scene",
  "characters": [{{"name": "...", "id": "ch0"}}],
  "attributes": ["unique attributes"],
  "interactions": ["unique interactions"],
  "scene_label": "place/setting label",
  "start_shot": {start_shot},
  "end_shot": {end_shot}
}}

Return ONLY the JSON."""


class SceneAnalyzer:
    """Analyze keyframes with VLM+LLM to generate scene graph data."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Lazy-init UniversalLLMClient."""
        if self._llm is None:
            try:
                import sys

                sys.path.insert(0, str(Cfg.SRC_DIR))
                from movierag.generation.universal_client import UniversalLLMClient

                self._llm = UniversalLLMClient()
                logger.info("  LLM client initialized")
            except Exception as e:
                logger.error(f"  Failed to init LLM client: {e}")
        return self._llm

    def analyze_movie(
        self,
        movie_id: str,
        meta: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Analyze all scenes of a movie by examining keyframes.

        Returns list of MovieGraphs-equivalent clip dicts.
        """
        logger.info(f"   Scene Analysis: {movie_id}")

        # Load keyframe index
        kf_index = self._load_keyframe_index(movie_id)
        if not kf_index:
            logger.warning(f"  No keyframe_index.json for {movie_id}")
            return []

        keyframes = kf_index.get("keyframes", [])
        title = (meta or {}).get("title", movie_id)

        # Group keyframes by scene_idx
        scene_groups: Dict[int, List[Dict]] = {}
        for kf in keyframes:
            si = kf["scene_idx"]
            scene_groups.setdefault(si, []).append(kf)

        clips = []
        total = len(scene_groups)
        llm = self._get_llm()

        for i, (si, kfs) in enumerate(sorted(scene_groups.items())):
            logger.info(
                f"  [{i + 1}/{total}] Analyzing scene {si} ({len(kfs)} keyframes)..."
            )

            # Pick the middle keyframe (img_idx=1) for VLM analysis
            best_kf = self._pick_representative(kfs)
            if not best_kf:
                continue

            start_sec = min(kf["timestamp_sec"] for kf in kfs)
            end_sec = max(kf["timestamp_sec"] for kf in kfs)

            # VLM analysis of keyframe
            vlm_result = self._analyze_keyframe_vlm(llm, best_kf["path"])

            if vlm_result:
                # Structure with LLM
                clip = self._structure_with_llm(
                    llm, vlm_result, movie_id, title, si, start_sec, end_sec, si, si
                )
            else:
                # Fallback: minimal clip
                clip = {
                    "situation": "unknown",
                    "description": f"Scene {si} of {title}",
                    "characters": [],
                    "attributes": [],
                    "interactions": [],
                    "scene_label": "",
                    "start_shot": si,
                    "end_shot": si,
                }

            clip["clip_id"] = f"{movie_id}_clip_{si:04d}"
            clip["auto_generated"] = True
            clips.append(clip)

        # Save
        if clips:
            out_path = self._save_scene_graph(movie_id, clips, title)
            logger.info(f"   Scene analysis: {len(clips)} clips → {out_path}")

        return clips

    def _analyze_keyframe_vlm(self, llm, image_path: str) -> Optional[Dict]:
        """Use VLM to analyze a single keyframe image."""
        if llm is None:
            return None

        img_path = Path(image_path)
        if not img_path.exists():
            return None

        try:
            response = llm.generate_vision_content(
                prompt=SCENE_ANALYSIS_PROMPT,
                image_path=str(img_path),
            )

            # Parse JSON from response
            text = response.text if hasattr(response, "text") else str(response)
            # Extract JSON from possible markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except Exception as e:
            logger.debug(f"  VLM analysis failed for {img_path.name}: {e}")
            return None

    def _structure_with_llm(
        self,
        llm,
        vlm_result,
        movie_id,
        title,
        scene_idx,
        start_sec,
        end_sec,
        start_shot,
        end_shot,
    ) -> Dict:
        """Use LLM to structure VLM output into MovieGraphs format."""
        if llm is None:
            return vlm_result or {}

        try:
            h1, m1, s1 = (
                int(start_sec // 3600),
                int((start_sec % 3600) // 60),
                int(start_sec % 60),
            )
            h2, m2, s2 = (
                int(end_sec // 3600),
                int((end_sec % 3600) // 60),
                int(end_sec % 60),
            )

            prompt = STRUCTURE_PROMPT.format(
                analyses=json.dumps(vlm_result, indent=2),
                title=title,
                movie_id=movie_id,
                scene_idx=scene_idx,
                start_time=f"{h1:02d}:{m1:02d}:{s1:02d}",
                end_time=f"{h2:02d}:{m2:02d}:{s2:02d}",
                start_shot=start_shot,
                end_shot=end_shot,
            )

            response = llm.generate_content(
                model="moonshotai/kimi-k2-instruct",
                contents=prompt,
            )

            text = response.text if hasattr(response, "text") else str(response)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except Exception as e:
            logger.debug(f"  LLM structuring failed: {e}")
            return vlm_result or {}

    @staticmethod
    def _pick_representative(kfs: List[Dict]) -> Optional[Dict]:
        """Pick best keyframe (prefer img_idx=1 = middle)."""
        priority = {1: 0, 0: 1, 2: 2}
        valid = [kf for kf in kfs if Path(kf.get("path", "")).exists()]
        if not valid:
            return None
        return sorted(valid, key=lambda x: priority.get(x.get("img_idx", 99), 99))[0]

    @staticmethod
    def _load_keyframe_index(movie_id: str) -> Dict:
        """Load keyframe_index.json for a movie."""
        for d in Cfg.KEYF_SEARCH_DIRS:
            idx_path = d / movie_id / "keyframe_index.json"
            if idx_path.exists():
                return json.loads(idx_path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _save_scene_graph(movie_id: str, clips: List[Dict], title: str) -> Path:
        """Save scene graph data as JSON."""
        out_dir = Cfg.UNIFIED_DATASET_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save as standalone scene graph file
        sg_path = out_dir / f"{movie_id}_scene_graph.json"
        sg_data = {
            "movie_id": movie_id,
            "title": title,
            "total_clips": len(clips),
            "auto_generated": True,
            "clips": clips,
        }
        sg_path.write_text(
            json.dumps(sg_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Also update unified dataset if exists
        ds_path = Cfg.UNIFIED_DATASET_JSON
        if ds_path.exists():
            try:
                ds = json.loads(ds_path.read_text(encoding="utf-8"))
                movies = ds.setdefault("movies", {})
                if movie_id not in movies:
                    movies[movie_id] = {
                        "imdb_id": movie_id,
                        "title": title,
                        "sources": {
                            "movienet": False,
                            "moviegraphs": False,
                            "auto": True,
                        },
                        "keyframes": [],
                        "clips": clips,
                    }
                else:
                    movies[movie_id]["clips"] = clips
                    movies[movie_id].setdefault("sources", {})["auto"] = True
                ds_path.write_text(
                    json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"  Could not update unified dataset: {e}")

        return sg_path
