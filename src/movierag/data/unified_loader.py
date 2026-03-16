"""
Loader for the Unified Dataset (Flow 1).
Reads the unified_dataset.json containing annotations, meta, scripts, and subtitles.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class UnifiedLoader:
    """
    Loads unified dataset from rawdata/unified_dataset.json.
    Provides structured textual documents representing movie plots, dialogs,
    and metadata for RAG indexing.
    """

    def __init__(self, data_root: str):
        self.data_root = Path(data_root).resolve()

        # Determine correct JSON path
        v2_path = self.data_root / "unified_dataset" / "movierag_dataset.json"
        v1_path = self.data_root / "rawdata" / "unified_dataset.json"

        self.json_path = v2_path if v2_path.exists() else v1_path
        self._cache = None

    def load(self) -> Dict[str, Any]:
        """Load the entire dataset (cached)."""
        if self._cache is not None:
            return self._cache

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"unified_dataset.json not found at {self.json_path}. "
                "Run scripts/build_flow1_dataset.py first."
            )

        logger.info(f"Loading unified dataset from {self.json_path}...")
        with open(self.json_path, "r", encoding="utf-8") as f:
            self._cache = json.load(f)

        movies = self._cache.get("movies", {})
        logger.info(f"Loaded {len(movies)} movies.")
        return self._cache

    def get_all_movie_ids(self) -> List[str]:
        """Get all available movie IDs."""
        data = self.load()
        return list(data.get("movies", {}).keys())

    def get_movie(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """Get movie object."""
        data = self.load()
        return data.get("movies", {}).get(imdb_id)

    def get_movie_info(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """Get basic info about a movie."""
        movie = self.get_movie(imdb_id)
        if not movie:
            return None

        return {
            "imdb_id": movie["imdb_id"],
            "title": movie.get("title", ""),
            "genres": movie.get("genres", []),
            "has_cast": bool(movie.get("cast", [])),
            "has_script": movie.get("sources", {}).get("script", False),
            "has_subtitle": movie.get("sources", {}).get("subtitle", False),
        }

    def get_textual_documents(self, imdb_id: str) -> List[Dict[str, Any]]:
        """
        Convert movie data into textual documents for RAG indexing.

        Each document contains:
        - movie_id: IMDB ID
        - clip_id / context_type: Identifier of the section
        - text: Full textual description
        - metadata: Additional structured info
        """
        movie = self.get_movie(imdb_id)
        if not movie:
            return []

        documents = []
        title = movie.get("title", "Unknown Title")

        # --- 1. Plot and Storyline Document ---
        plot_text_parts = []
        plot = movie.get("plot", "")
        overview = movie.get("overview", "")
        storyline = movie.get("storyline", "")

        if plot:
            plot_text_parts.append(f"Plot: {plot}")
        if overview:
            plot_text_parts.append(f"Overview: {overview}")
        if storyline:
            plot_text_parts.append(f"Storyline: {storyline}")

        if plot_text_parts:
            full_text = f"Movie Title: {title}.\n" + "\n".join(plot_text_parts)
            documents.append(
                {
                    "movie_id": imdb_id,
                    "clip_id": "plot_overview",
                    "text": full_text,
                    "metadata": {
                        "category": "meta",
                        "title": title,
                        "genres": movie.get("genres", []),
                    },
                }
            )

        # --- 2. Cast and Character Document ---
        cast_list = movie.get("cast", [])
        if cast_list:
            cast_text_parts = [f"Cast for {title}:"]
            for actor in cast_list:
                name = actor.get("name", "")
                char = actor.get("character", "")
                if name and char:
                    cast_text_parts.append(f"- Actor {name} plays character {char}.")
                elif name:
                    cast_text_parts.append(f"- Actor {name}.")

            full_text = "\n".join(cast_text_parts)
            documents.append(
                {
                    "movie_id": imdb_id,
                    "clip_id": "cast_list",
                    "text": full_text,
                    "metadata": {"category": "cast", "title": title},
                }
            )

        # --- 3. Subtitles Document chunks ---
        subtitles = movie.get("subtitles", [])
        if subtitles:
            chunk_size = 30  # Group 30 subtitle lines into a chunk
            for i in range(0, len(subtitles), chunk_size):
                chunk_subs = subtitles[i : i + chunk_size]
                text_lines = []
                start_time = chunk_subs[0].get("timestamps", "")
                end_time = chunk_subs[-1].get("timestamps", "")

                for s in chunk_subs:
                    text_lines.append(s.get("text", ""))

                full_text = f"Subtitles for {title} (Dialog Context):\n" + " ".join(
                    text_lines
                )

                documents.append(
                    {
                        "movie_id": imdb_id,
                        "clip_id": f"subtitle_chunk_{i}",
                        "text": full_text,
                        "metadata": {
                            "category": "subtitle",
                            "start_time": start_time,
                            "end_time": end_time,
                        },
                    }
                )

        # --- 4. Script Document chunks ---
        script_text = movie.get("script", "")
        if script_text:
            # Simple chunking by paragraph or character count
            # A script is large, we chunk roughly 2000 chars
            chunk_len = 2000
            for i in range(0, len(script_text), chunk_len):
                text_chunk = script_text[i : i + chunk_len]
                full_text = f"Script Scene Context for {title}:\n{text_chunk}"

                documents.append(
                    {
                        "movie_id": imdb_id,
                        "clip_id": f"script_chunk_{i}",
                        "text": full_text,
                        "metadata": {
                            "category": "script",
                        },
                    }
                )

        # --- 5. MovieGraph (SceneRAG) Document chunks ---
        moviegraph = movie.get("moviegraph", {})
        if moviegraph:
            # MovieGraph typically contains nodes (characters, objects), edges (actions, relations), and scenes.

            # Global relationships
            relations = moviegraph.get("relationships", [])
            if relations:
                rel_parts = [f"Character Relationships in {title}:"]
                for rel in relations:
                    p1 = rel.get("person1", "")
                    p2 = rel.get("person2", "")
                    r_type = rel.get("relation", "")
                    if p1 and p2 and r_type:
                        rel_parts.append(f"- {p1} is {r_type} to {p2}.")

                documents.append(
                    {
                        "movie_id": imdb_id,
                        "clip_id": "moviegraph_relations",
                        "text": "\n".join(rel_parts),
                        "metadata": {"category": "moviegraph", "title": title},
                    }
                )

            # Scene-level actions (SceneRAG Multi-hop context)
            scenes = moviegraph.get("scenes", [])
            if scenes:
                for idx, scene in enumerate(scenes):
                    scene_id = scene.get("scene_id", f"scene_{idx}")
                    actions = scene.get("actions", [])
                    if actions:
                        act_parts = [f"Scene {scene_id} actions in {title}:"]
                        for act in actions:
                            subj = act.get("subject", "")
                            verb = act.get("verb", "")
                            obj = act.get("object", "")
                            if subj and verb:
                                act_parts.append(
                                    f"- {subj} {verb} {obj}."
                                    if obj
                                    else f"- {subj} {verb}."
                                )

                        documents.append(
                            {
                                "movie_id": imdb_id,
                                "clip_id": f"moviegraph_{scene_id}",
                                "text": "\n".join(act_parts),
                                "metadata": {
                                    "category": "moviegraph",
                                    "scene_id": scene_id,
                                    "title": title,
                                },
                            }
                        )

                documents.append(
                    {
                        "movie_id": imdb_id,
                        "clip_id": f"script_chunk_{i}",
                        "text": full_text,
                        "metadata": {"category": "script_scene", "offset": i},
                    }
                )

        return documents

    def get_all_textual_documents(self) -> List[Dict[str, Any]]:
        """Get textual documents for all movies."""
        all_docs = []
        for movie_id in self.get_all_movie_ids():
            docs = self.get_textual_documents(movie_id)
            all_docs.extend(docs)
        return all_docs
