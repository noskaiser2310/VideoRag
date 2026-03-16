"""
Loader for MovieGraphs dataset (Pickle version).

MovieGraphs provides rich scene-level annotations including:
- Situation descriptions
- Scene labels
- Character interactions
- Relationships between entities
- Graph-based story representation
"""

import pickle
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MovieGraphLoader:
    """
    Loads MovieGraphs data from all_movies.pkl.

    The pickle file contains MovieGraph objects that require GraphClasses.py
    from the py3loader_new directory to deserialize properly.
    """

    def __init__(self, data_root: str):
        """
        Args:
            data_root: Path to MovieGraphs_repo directory
        """
        self.data_root = Path(data_root).resolve()
        self.pkl_path = None
        self.py3loader_path = None
        self._cache = None
        self._setup_paths()

    def _setup_paths(self):
        """Find the pickle file and py3loader directory."""
        # Look for all_movies.pkl
        direct_path = self.data_root / "all_movies.pkl"
        if direct_path.exists():
            self.pkl_path = direct_path
        else:
            # Search in subdirectories
            candidates = list(self.data_root.rglob("all_movies.pkl"))
            if candidates:
                self.pkl_path = candidates[0]

        if self.pkl_path:
            # py3loader_new should be in same directory as pkl
            self.py3loader_path = self.pkl_path.parent
            logger.info(f"Found MovieGraphs at: {self.pkl_path}")
            logger.info(f"py3loader path: {self.py3loader_path}")

    def _ensure_imports(self):
        """Add py3loader_new to path for GraphClasses import."""
        if self.py3loader_path and str(self.py3loader_path) not in sys.path:
            sys.path.insert(0, str(self.py3loader_path))
            logger.debug(f"Added {self.py3loader_path} to sys.path")

    def load(self) -> Dict[str, Any]:
        """Load the entire dataset (cached)."""
        if self._cache is not None:
            return self._cache

        if not self.pkl_path or not self.pkl_path.exists():
            raise FileNotFoundError(
                f"all_movies.pkl not found in {self.data_root}. "
                "Download from MovieGraphs repository."
            )

        # Ensure GraphClasses can be imported
        self._ensure_imports()

        logger.info(f"Loading MovieGraphs from {self.pkl_path}...")
        with open(self.pkl_path, "rb") as f:
            # encoding='latin1' is crucial for python 2 pickles loaded in py3
            self._cache = pickle.load(f, encoding="latin1")

        logger.info(f"Loaded {len(self._cache)} movies.")
        return self._cache

    def get_all_movie_ids(self) -> List[str]:
        """Get all available movie IDs (IMDB format)."""
        data = self.load()
        return list(data.keys())

    def get_movie(self, imdb_id: str) -> Optional[Any]:
        """Get MovieGraph object for a specific movie."""
        data = self.load()
        return data.get(imdb_id)

    def get_movie_info(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """Get basic info about a movie."""
        mg = self.get_movie(imdb_id)
        if not mg:
            return None

        return {
            "imdb_id": imdb_id,
            "imdb_key": getattr(mg, "imdb_key", imdb_id),
            "num_clips": len(mg.clip_graphs) if hasattr(mg, "clip_graphs") else 0,
            "has_castlist": hasattr(mg, "castlist"),
        }

    def get_all_clips(self, imdb_id: str) -> List[Dict[str, Any]]:
        """
        Get all clips for a movie with their graph information.

        Returns list of dicts with:
        - clip_id: Clip identifier
        - situation: Scene situation description
        - scene_label: Scene type label
        - description: Text description
        - characters: List of character names
        - num_nodes: Number of nodes in graph
        - num_edges: Number of edges in graph
        """
        mg = self.get_movie(imdb_id)
        if not mg:
            return []

        clips = []
        clip_graphs = mg.clip_graphs

        # Handle both dict and OrderedDict
        if hasattr(clip_graphs, "items"):
            items = clip_graphs.items()
        else:
            items = enumerate(clip_graphs)

        for clip_id, cg in items:
            clip_info = {
                "movie_id": imdb_id,
                "clip_id": str(clip_id),
                "situation": getattr(cg, "situation", ""),
                "scene_label": getattr(cg, "scene_label", ""),
                "description": getattr(cg, "description", ""),
            }

            # Extract characters from graph
            if hasattr(cg, "G"):
                graph = cg.G
                characters = []
                for n in graph.nodes():
                    node_data = (
                        graph.node[n] if hasattr(graph, "node") else graph.nodes[n]
                    )
                    if node_data.get("type") == "entity":
                        characters.append(node_data.get("name", "Unknown"))

                clip_info["characters"] = characters
                clip_info["num_nodes"] = len(graph.nodes())
                clip_info["num_edges"] = len(graph.edges())
            else:
                clip_info["characters"] = []
                clip_info["num_nodes"] = 0
                clip_info["num_edges"] = 0

            clips.append(clip_info)

        return clips

    def get_textual_documents(self, imdb_id: str) -> List[Dict[str, Any]]:
        """
        Convert movie clips into textual documents for RAG indexing.

        Each document contains:
        - movie_id: IMDB ID
        - clip_id: Clip identifier
        - text: Full textual description
        - metadata: Additional structured info
        """
        mg = self.get_movie(imdb_id)
        if not mg:
            return []

        documents = []
        clip_graphs = mg.clip_graphs

        if hasattr(clip_graphs, "items"):
            items = clip_graphs.items()
        else:
            items = enumerate(clip_graphs)

        for clip_id, cg in items:
            text_parts = []

            # Scene context
            situation = getattr(cg, "situation", "")
            scene_label = getattr(cg, "scene_label", "")
            description = getattr(cg, "description", "")

            if scene_label:
                text_parts.append(f"Scene: {scene_label}.")
            if situation:
                text_parts.append(f"Situation: {situation}.")

            # Extract from graph
            characters = []
            relationships = []
            interactions = []
            attributes = []

            if hasattr(cg, "G"):
                graph = cg.G

                # Get all nodes by type
                for n in graph.nodes():
                    node_data = (
                        graph.node[n] if hasattr(graph, "node") else graph.nodes[n]
                    )
                    node_type = node_data.get("type", "")
                    node_name = node_data.get("name", "")

                    if node_type == "entity":
                        characters.append(node_name)
                    elif node_type == "relationship":
                        relationships.append(node_name)
                    elif node_type == "interaction":
                        interactions.append(node_name)
                    elif node_type == "attribute":
                        attributes.append(node_name)

                # Build triplets (entity -> relationship/interaction -> entity)
                triplets = []
                if hasattr(cg, "find_all_triplets"):
                    for t_type in ["relationship", "interaction"]:
                        try:
                            trips = cg.find_all_triplets(
                                int_or_rel=t_type, return_names=True
                            )
                            triplets.extend(trips)
                        except:
                            pass

                if triplets:
                    for e1, rel, e2 in triplets:
                        text_parts.append(f"{e1} {rel} {e2}.")

            # Add characters
            if characters:
                text_parts.append(f"Characters: {', '.join(set(characters))}.")

            # Add description
            if description:
                text_parts.append(description)

            full_text = " ".join(text_parts)

            if full_text.strip():
                documents.append(
                    {
                        "movie_id": imdb_id,
                        "clip_id": str(clip_id),
                        "text": full_text,
                        "metadata": {
                            "scene_label": scene_label,
                            "situation": situation,
                            "characters": list(set(characters)),
                            "num_relationships": len(relationships),
                            "num_interactions": len(interactions),
                        },
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
