"""
MovieGraphs Dataset Loader

Handles loading of MovieGraphs dataset including:
- Graph annotations (situations, interactions, mental states)
- Character information
- Clip metadata

Reference: https://arxiv.org/pdf/1712.06761
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """An interaction between characters."""

    source_character: str
    target_character: str
    interaction_type: str
    description: str
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class MentalState:
    """Mental state of a character."""

    character: str
    state_type: str  # e.g., "emotion", "attitude"
    value: str  # e.g., "angry", "sad", "hopeful"
    confidence: float = 1.0


@dataclass
class Situation:
    """A situation (clip) in MovieGraphs."""

    clip_id: str
    movie_id: str
    situation_label: str
    scene_label: str
    description: str
    start_time: float
    end_time: float
    characters: List[str] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    mental_states: List[MentalState] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MovieGraph:
    """A movie with all its graph annotations."""

    movie_id: str
    title: str
    situations: List[Situation] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)


class MovieGraphsLoader:
    """
    Loader for MovieGraphs dataset.

    Expected directory structure:
    MovieGraphs_repo/
     all_movies.pkl          # Main graph annotations
     split.json              # Train/val/test splits
     GraphClasses.py         # Graph class definitions
     videos/                 # Video clips (if available)
    """

    def __init__(self, data_root: str):
        """
        Initialize the MovieGraphs loader.

        Args:
            data_root: Path to MovieGraphs data directory
        """
        self.data_root = Path(data_root)

        # Cache
        self._graphs_cache: Optional[Dict[str, Any]] = None
        self._splits_cache: Optional[Dict[str, List[str]]] = None

        logger.info(f"MovieGraphsLoader initialized with root: {self.data_root}")

    def _load_graphs(self) -> Dict[str, Any]:
        """Load the main graph annotations."""
        if self._graphs_cache is not None:
            return self._graphs_cache

        pkl_file = self.data_root / "all_movies.pkl"
        if pkl_file.exists():
            try:
                with open(pkl_file, "rb") as f:
                    self._graphs_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._graphs_cache)} movie graphs")
            except Exception as e:
                logger.error(f"Failed to load graphs: {e}")
                self._graphs_cache = {}
        else:
            logger.warning(f"Graph file not found: {pkl_file}")
            self._graphs_cache = {}

        return self._graphs_cache

    def _load_splits(self) -> Dict[str, List[str]]:
        """Load train/val/test splits."""
        if self._splits_cache is not None:
            return self._splits_cache

        split_file = self.data_root / "split.json"
        if split_file.exists():
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    self._splits_cache = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load splits: {e}")
                self._splits_cache = {"train": [], "val": [], "test": []}
        else:
            logger.warning(f"Split file not found: {split_file}")
            self._splits_cache = {"train": [], "val": [], "test": []}

        return self._splits_cache

    def get_all_movie_ids(self) -> List[str]:
        """Get list of all available movie IDs."""
        graphs = self._load_graphs()
        return list(graphs.keys())

    def get_movie_graph(self, movie_id: str) -> Optional[MovieGraph]:
        """
        Load a movie with all its graph annotations.

        Args:
            movie_id: Movie identifier

        Returns:
            MovieGraph object or None if not found
        """
        graphs = self._load_graphs()

        if movie_id not in graphs:
            logger.warning(f"Movie not found: {movie_id}")
            return None

        raw_data = graphs[movie_id]

        # Parse the raw graph data into our dataclass structure
        movie_graph = MovieGraph(
            movie_id=movie_id, title=raw_data.get("title", movie_id)
        )

        # Extract unique characters
        all_characters = set()

        # Parse clips/situations
        for clip_data in raw_data.get("clips", []):
            situation = self._parse_situation(clip_data, movie_id)
            movie_graph.situations.append(situation)
            all_characters.update(situation.characters)

        movie_graph.characters = list(all_characters)

        return movie_graph

    def _parse_situation(self, clip_data: Dict, movie_id: str) -> Situation:
        """Parse a clip/situation from raw data."""
        situation = Situation(
            clip_id=clip_data.get("clip_id", ""),
            movie_id=movie_id,
            situation_label=clip_data.get("situation_label", ""),
            scene_label=clip_data.get("scene_label", ""),
            description=clip_data.get("description", ""),
            start_time=clip_data.get("start_time", 0.0),
            end_time=clip_data.get("end_time", 0.0),
        )

        # Parse characters
        situation.characters = clip_data.get("characters", [])

        # Parse interactions
        for interaction_data in clip_data.get("interactions", []):
            interaction = Interaction(
                source_character=interaction_data.get("source", ""),
                target_character=interaction_data.get("target", ""),
                interaction_type=interaction_data.get("type", ""),
                description=interaction_data.get("description", ""),
                start_time=interaction_data.get("start_time", 0.0),
                end_time=interaction_data.get("end_time", 0.0),
            )
            situation.interactions.append(interaction)

        # Parse mental states
        for state_data in clip_data.get("mental_states", []):
            mental_state = MentalState(
                character=state_data.get("character", ""),
                state_type=state_data.get("type", ""),
                value=state_data.get("value", ""),
                confidence=state_data.get("confidence", 1.0),
            )
            situation.mental_states.append(mental_state)

        return situation

    def get_situations_by_split(self, split: str = "train") -> Iterator[Situation]:
        """
        Iterate over situations in a specific split.

        Args:
            split: One of "train", "val", "test"

        Yields:
            Situation objects
        """
        splits = self._load_splits()
        movie_ids = splits.get(split, [])

        for movie_id in movie_ids:
            movie_graph = self.get_movie_graph(movie_id)
            if movie_graph:
                for situation in movie_graph.situations:
                    yield situation

    def search_by_interaction(self, interaction_type: str) -> List[Situation]:
        """
        Search for situations containing a specific interaction type.

        Args:
            interaction_type: Type of interaction to search for

        Returns:
            List of matching situations
        """
        results = []

        for movie_id in self.get_all_movie_ids():
            movie_graph = self.get_movie_graph(movie_id)
            if not movie_graph:
                continue

            for situation in movie_graph.situations:
                for interaction in situation.interactions:
                    if interaction_type.lower() in interaction.interaction_type.lower():
                        results.append(situation)
                        break

        return results

    def search_by_mental_state(self, emotion: str) -> List[Situation]:
        """
        Search for situations where characters have a specific mental state.

        Args:
            emotion: Mental state to search for (e.g., "angry", "sad")

        Returns:
            List of matching situations
        """
        results = []

        for movie_id in self.get_all_movie_ids():
            movie_graph = self.get_movie_graph(movie_id)
            if not movie_graph:
                continue

            for situation in movie_graph.situations:
                for mental_state in situation.mental_states:
                    if emotion.lower() in mental_state.value.lower():
                        results.append(situation)
                        break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        graphs = self._load_graphs()
        splits = self._load_splits()

        total_situations = 0
        total_interactions = 0
        total_mental_states = 0

        for movie_id in graphs.keys():
            movie_graph = self.get_movie_graph(movie_id)
            if movie_graph:
                total_situations += len(movie_graph.situations)
                for situation in movie_graph.situations:
                    total_interactions += len(situation.interactions)
                    total_mental_states += len(situation.mental_states)

        return {
            "num_movies": len(graphs),
            "num_train": len(splits.get("train", [])),
            "num_val": len(splits.get("val", [])),
            "num_test": len(splits.get("test", [])),
            "total_situations": total_situations,
            "total_interactions": total_interactions,
            "total_mental_states": total_mental_states,
            "data_root": str(self.data_root),
        }
