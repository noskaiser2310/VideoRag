"""
MovieNet Dataset Loader

Handles loading of MovieNet dataset structure including:
- Movie metadata
- Scene boundaries
- Shot information
- Character annotations
- Cinematic style tags

Reference: https://movienet.github.io/
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator
from dataclasses import dataclass, field
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Character:
    """Character instance in a frame."""

    movie_id: str
    frame_id: int
    bbox: List[float]  # [x, y, w, h]
    identity: str
    actor: Optional[str] = None


@dataclass
class Shot:
    """A shot within a scene."""

    shot_id: str
    movie_id: str
    start_frame: int
    end_frame: int
    keyframe_path: Optional[str] = None
    view_scale: Optional[str] = None  # CU, MS, LS, etc.
    camera_motion: Optional[str] = None  # static, pan, tilt, zoom


@dataclass
class Scene:
    """A scene containing multiple shots."""

    scene_id: str
    movie_id: str
    start_frame: int
    end_frame: int
    description: Optional[str] = None
    shots: List[Shot] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    action_tags: List[str] = field(default_factory=list)
    place_tags: List[str] = field(default_factory=list)


@dataclass
class Movie:
    """A movie with all its annotations."""

    movie_id: str
    title: str
    year: Optional[int] = None
    genres: List[str] = field(default_factory=list)
    cast: List[str] = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
    fps: float = 24.0
    duration_seconds: float = 0.0


class MovieNetLoader:
    """
    Loader for MovieNet dataset.

    Expected directory structure:
    movienet/
     annotations/
        character/
        scene/
        action/
        place/
        style/
     meta/
        movie_info.json
     keyframes/
        {movie_id}/
            shot_{xxxx}.jpg
     videos/
         {movie_id}.mp4
    """

    def __init__(
        self,
        data_root: str,
        use_sample: bool = False,
        keyframes_dir: str = None,
        shot_txt_dir: str = None,
        meta_dir: str = None,
        annotation_dir: str = None,
        raw_videos_dir: str = None,
    ):
        """
        Initialize the MovieNet loader.

        Args:
            data_root: Path to MovieNet data directory (data/movienet)
            use_sample: If True, use sample data from movienet-tools for testing
            keyframes_dir: Override path to shot_keyf directory
            shot_txt_dir: Override path to shot_txt directory
            meta_dir: Override path to meta JSON directory
            annotation_dir: Override path to annotation JSON directory
            raw_videos_dir: Override path to raw video directory
        """
        self.data_root = Path(data_root)
        self.use_sample = use_sample

        # Define paths (can be overridden via args)
        self.keyframes_dir = (
            Path(keyframes_dir) if keyframes_dir else self.data_root / "shot_keyf"
        )
        self.shot_txt_dir = (
            Path(shot_txt_dir) if shot_txt_dir else self.data_root / "shot_txt"
        )
        self.meta_dir = Path(meta_dir) if meta_dir else self.data_root / "meta"
        self.annotations_dir = (
            Path(annotation_dir) if annotation_dir else self.data_root / "annotation"
        )
        self.videos_dir = (
            Path(raw_videos_dir) if raw_videos_dir else self.data_root / "videos"
        )

        # Cache
        self._movies_cache: Dict[str, Movie] = {}
        self._keyframes_cache: Dict[str, List[str]] = {}
        self._annotations_cache: Dict[str, List[Dict]] = {}

        logger.info(
            f"MovieNetLoader initialized: keyframes={self.keyframes_dir}, meta={self.meta_dir}"
        )

    def get_all_movie_ids(self) -> List[str]:
        """Get list of all available movie IDs."""
        if self.use_sample:
            # For sample data, infer from keyframe directories
            if self.keyframes_dir.exists():
                return [d.name for d in self.keyframes_dir.iterdir() if d.is_dir()]
            return []

        # Try loading from meta file
        meta_file = self.meta_dir / "movie_info.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data.keys())

        # Fallback: list from keyframes directory
        if self.keyframes_dir.exists():
            return [d.name for d in self.keyframes_dir.iterdir() if d.is_dir()]

        return []

    def get_movie(self, movie_id: str) -> Optional[Movie]:
        """Load a movie with all its annotations."""
        if movie_id in self._movies_cache:
            return self._movies_cache[movie_id]

        movie = Movie(movie_id=movie_id, title=movie_id)

        if not self.use_sample:
            # Load metadata
            self._load_movie_metadata(movie)
            # Load scenes
            self._load_scenes(movie)
            # Load characters
            self._load_characters(movie)

        self._movies_cache[movie_id] = movie
        return movie

    def _load_movie_metadata(self, movie: Movie) -> None:
        """Load movie metadata from per-movie JSON files."""
        # Try per-movie JSON first (e.g., meta/tt0097576.json)
        meta_file = self.meta_dir / f"{movie.movie_id}.json"
        if not meta_file.exists():
            # Fallback to legacy single-file format
            meta_file = self.meta_dir / "movie_info.json"

        if not meta_file.exists():
            return

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Per-movie JSON is the movie object directly
            if "imdb_id" in data:
                meta = data
            elif movie.movie_id in data:
                meta = data[movie.movie_id]
            else:
                return

            movie.title = meta.get("title", movie.movie_id)
            movie.year = meta.get("year")
            movie.genres = meta.get("genres", [])

            # Parse cast list
            cast_list = meta.get("cast", [])
            if cast_list and isinstance(cast_list[0], dict):
                movie.cast = [
                    f"{c.get('name', '')} as {c.get('character', '')}"
                    for c in cast_list
                    if c.get("name")
                ]
            else:
                movie.cast = cast_list

            # Store the full FPS if available from version runtime
            versions = meta.get("version", [])
            if versions:
                runtime_str = versions[0].get("runtime", "")
                # Parse "127 min" -> estimate FPS as 24.0 default
                import re

                match = re.search(r"(\d+)", runtime_str)
                if match:
                    movie.duration_seconds = int(match.group(1)) * 60

        except Exception as e:
            logger.warning(f"Failed to load metadata for {movie.movie_id}: {e}")

    def _load_annotations(self, movie_id: str) -> List[Dict[str, Any]]:
        """Load character annotations (bounding boxes + actor PIDs) from annotation JSON."""
        if movie_id in self._annotations_cache:
            return self._annotations_cache[movie_id]

        ann_file = self.annotations_dir / f"{movie_id}.json"
        if not ann_file.exists():
            self._annotations_cache[movie_id] = []
            return []

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cast_annotations = data.get("cast", [])
            self._annotations_cache[movie_id] = cast_annotations
            logger.info(
                f"Loaded {len(cast_annotations)} character annotations for {movie_id}"
            )
            return cast_annotations

        except Exception as e:
            logger.warning(f"Failed to load annotations for {movie_id}: {e}")
            self._annotations_cache[movie_id] = []
            return []

    def get_characters_in_shot(self, movie_id: str, shot_idx: int) -> List[str]:
        """
        Get unique character/actor PIDs that appear in a given shot.

        Args:
            movie_id: IMDB ID
            shot_idx: Shot index number

        Returns:
            List of unique PIDs (e.g., ['nm0000148', 'nm0000125', 'others'])
        """
        annotations = self._load_annotations(movie_id)
        pids = set()
        for ann in annotations:
            if ann.get("shot_idx") == shot_idx:
                pid = ann.get("pid", "others")
                pids.add(pid)
        return list(pids)

    def _load_shots_from_txt(self, movie_id: str) -> List[Shot]:
        """
        Load shot boundaries from shot_txt file (MovieNet structure).
        Format: start_frame end_frame keyframe_idx1 keyframe_idx2 keyframe_idx3
        """
        shot_file = self.shot_txt_dir / f"{movie_id}.txt"

        if not shot_file.exists():
            logger.warning(f"Shot info not found for {movie_id}: {shot_file}")
            return []

        shots = []
        try:
            with open(shot_file, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = list(map(int, line.strip().split()))
                if len(parts) < 2:
                    continue

                start_frame = parts[0]
                end_frame = parts[1]

                # Create Shot object
                shot = Shot(
                    shot_id=f"{movie_id}_shot_{i:04d}",
                    movie_id=movie_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                shots.append(shot)

            logger.info(f"Loaded {len(shots)} shots for {movie_id} from txt")
            return shots

        except Exception as e:
            logger.error(f"Failed to parse shot txt for {movie_id}: {e}")
            return []

    def _load_scenes(self, movie: Movie) -> None:
        """Load scene boundaries."""
        # PRIORITIZE: Load shots from txt first (standard MovieNet structure)
        shots = self._load_shots_from_txt(movie.movie_id)
        if shots:
            # Create a default scene containing these shots if no scene info exists
            scene = Scene(
                scene_id=f"{movie.movie_id}_scene_full",
                movie_id=movie.movie_id,
                start_frame=shots[0].start_frame,
                end_frame=shots[-1].end_frame,
                shots=shots,
            )
            movie.scenes.append(scene)
            return

        # Fallback to JSON scene annotations if txt not found (old logic)
        scene_file = self.annotations_dir / "scene" / f"{movie.movie_id}.json"
        if not scene_file.exists():
            return

        try:
            with open(scene_file, "r", encoding="utf-8") as f:
                scene_data = json.load(f)

            for i, boundary in enumerate(scene_data.get("boundaries", [])):
                scene = Scene(
                    scene_id=f"{movie.movie_id}_scene_{i:04d}",
                    movie_id=movie.movie_id,
                    start_frame=boundary.get("start", 0),
                    end_frame=boundary.get("end", 0),
                    description=boundary.get("description"),
                )
                movie.scenes.append(scene)
        except Exception as e:
            logger.warning(f"Failed to load scenes for {movie.movie_id}: {e}")

    def get_keyframes(self, movie_id: str) -> List[str]:
        """
        Get all keyframe paths for a movie.

        Returns:
            List of absolute paths to keyframe images
        """
        if movie_id in self._keyframes_cache:
            return self._keyframes_cache[movie_id]

        keyframe_dir = self.keyframes_dir / movie_id
        if not keyframe_dir.exists():
            # Try without movie_id subdirectory (for sample data)
            keyframe_dir = self.keyframes_dir

        if not keyframe_dir.exists():
            logger.warning(f"Keyframe directory not found: {keyframe_dir}")
            return []

        # Find all image files
        keyframes = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            keyframes.extend([str(p) for p in keyframe_dir.glob(ext)])

        # Sort by name to maintain order
        keyframes.sort()

        self._keyframes_cache[movie_id] = keyframes
        logger.info(f"Found {len(keyframes)} keyframes for movie: {movie_id}")

        return keyframes

    def get_all_keyframes(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all keyframes in the dataset.

        Yields:
            Dict with 'movie_id', 'keyframe_path', 'keyframe_id'
        """
        if self.use_sample:
            # Iterate over known sample movies (test1, test2)
            sample_movies = [d.name for d in self.keyframes_dir.iterdir() if d.is_dir()]

            for movie_id in sample_movies:
                # Pre-load shots
                shots = self._load_shots_from_txt(movie_id)
                shot_map = (
                    {int(s.shot_id.split("_")[-1]): s for s in shots} if shots else {}
                )

                # Sample FPS (usually 24 or 25, let's assume 24)
                fps = 24.0

                keyframe_dir = self.keyframes_dir / movie_id
                for img_path in keyframe_dir.glob("*.jpg"):
                    keyframe_id = img_path.stem  # e.g. shot_0000

                    # Extract shot index
                    shot_idx = -1
                    try:
                        if "shot_" in keyframe_id:
                            shot_idx = int(keyframe_id.split("_")[-1])
                    except ValueError:
                        pass

                    item = {
                        "movie_id": movie_id,
                        "keyframe_path": str(img_path.absolute()),
                        "keyframe_id": keyframe_id,
                    }

                    # Add Shot Metadata if matches
                    if shot_idx in shot_map:
                        shot = shot_map[shot_idx]
                        item["shot_id"] = shot.shot_id
                        item["start_frame"] = shot.start_frame
                        item["end_frame"] = shot.end_frame
                        item["timestamp"] = round(shot.start_frame / fps, 2)
                        item["timestamp_end"] = round(shot.end_frame / fps, 2)

                    yield item

            logger.info(f"Processed sample movies: {sample_movies}")

        else:
            for movie_id in self.get_all_movie_ids():
                # Pre-load shots for efficiency
                shots = self._load_shots_from_txt(movie_id)
                shot_map = (
                    {int(s.shot_id.split("_")[-1]): s for s in shots} if shots else {}
                )

                # Default FPS 24.0 (should load from meta if available)
                movie = self.get_movie(movie_id)
                fps = movie.fps if movie else 24.0

                for keyframe_path in self.get_keyframes(movie_id):
                    keyframe_id = Path(keyframe_path).stem

                    # Extract shot index from filename (shot_0000 -> 0)
                    shot_idx = -1
                    try:
                        if "shot_" in keyframe_id:
                            shot_idx = int(keyframe_id.split("_")[-1])
                    except ValueError:
                        pass

                    item = {
                        "movie_id": movie_id,
                        "keyframe_path": keyframe_path,
                        "keyframe_id": keyframe_id,
                    }

                    # Add Shot Metadata if matches
                    if shot_idx in shot_map:
                        shot = shot_map[shot_idx]
                        item["shot_id"] = shot.shot_id
                        item["start_frame"] = shot.start_frame
                        item["end_frame"] = shot.end_frame
                        # Simple timestamp (seconds)
                        item["timestamp"] = round(shot.start_frame / fps, 2)
                        item["timestamp_end"] = round(shot.end_frame / fps, 2)

                    yield item

    def extract_keyframes_from_video(
        self,
        video_path: str,
        output_dir: str,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> List[str]:
        """
        Extract keyframes from a video file.

        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract

        Returns:
            List of paths to extracted keyframes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(video_fps / fps) if fps > 0 else 1

        extracted_paths = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_paths.append(str(frame_path))
                saved_count += 1

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {saved_count} keyframes from {video_path}")

        return extracted_paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        movie_ids = self.get_all_movie_ids()
        total_keyframes = sum(len(self.get_keyframes(mid)) for mid in movie_ids)

        return {
            "num_movies": len(movie_ids),
            "total_keyframes": total_keyframes,
            "data_root": str(self.data_root),
            "use_sample": self.use_sample,
        }
