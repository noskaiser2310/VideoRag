"""
Visual Indexer for Flow 1: Visual Search.

Combines CLIP embeddings with FAISS for efficient similarity search
of movie frames and shots.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from PIL import Image
import subprocess

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result."""

    id: str
    path: str
    movie_id: str
    score: float
    metadata: Dict[str, Any]


class VisualIndexer:
    """
    Visual indexer for movie frame search.

    Uses CLIP for encoding and FAISS for efficient similarity search.
    Supports both text-to-image and image-to-image search.
    """

    def __init__(
        self,
        index_dir: str,
        index_name: str = "visual_index",
        encoder: Optional[CLIPEncoder] = None,
    ):
        """
        Initialize the visual indexer.

        Args:
            index_dir: Directory to store/load index files
            index_name: Base name for index files
            encoder: CLIPEncoder instance (creates new one if None)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_name = index_name
        self.index_path = self.index_dir / f"{index_name}.faiss"
        self.metadata_path = self.index_dir / f"{index_name}_metadata.json"

        # Hierarchical Scene L1 paths
        self.scene_index_path = self.index_dir / f"{index_name}_scenes.faiss"
        self.scene_metadata_path = self.index_dir / f"{index_name}_scenes_meta.json"

        # Initialize encoder
        self.encoder = encoder or CLIPEncoder()

        # Index state
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._is_loaded = False

        self._scene_index = None
        self._scene_metadata: List[Dict[str, Any]] = []
        self._scene_is_loaded = False

    def build_index(
        self,
        items: List[Dict[str, Any]],
        id_key: str = "keyframe_id",
        path_key: str = "keyframe_path",
        movie_id_key: str = "movie_id",
    ) -> None:
        """
        Build the visual index from a list of items.

        Args:
            items: List of dicts with image paths and metadata
            id_key: Key for item ID in the dict
            path_key: Key for image path in the dict
            movie_id_key: Key for movie ID in the dict
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )

        logger.info(f"Building index from {len(items)} items...")

        # Extract paths and metadata
        image_paths = []
        self._metadata = []

        for item in items:
            image_paths.append(item[path_key])
            self._metadata.append(
                {
                    "id": item[id_key],
                    "path": item[path_key],
                    "movie_id": item.get(movie_id_key, "unknown"),
                    **{
                        k: v
                        for k, v in item.items()
                        if k not in [id_key, path_key, movie_id_key]
                    },
                }
            )

        # Encode all images
        logger.info("Encoding images with CLIP...")
        embeddings = self.encoder.encode_images(image_paths, normalize=True)

        if len(embeddings) == 0:
            logger.error("No images could be encoded")
            return

        # Build FAISS index
        dim = embeddings.shape[1]
        logger.info(
            f"Building FAISS index with {len(embeddings)} vectors of dimension {dim}"
        )

        # Use IndexFlatIP for cosine similarity (vectors are normalized)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        self._is_loaded = True

        # Save index
        self.save()

        logger.info(f"Index built successfully with {self._index.ntotal} vectors")

    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self._index is None:
            logger.warning("No index to save")
            return

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")

        # Save metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {self.metadata_path}")

    def load(self) -> bool:
        """
        Load the index and metadata from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_path.exists():
            logger.warning(f"Index file not found: {self.index_path}")
            return False

        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return False

        try:
            # Load FAISS index
            self._index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")

            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self._is_loaded = True

            # Optionally load Hierarchical Scene Level Index (L1)
            if self.scene_index_path.exists() and self.scene_metadata_path.exists():
                try:
                    self._scene_index = faiss.read_index(str(self.scene_index_path))
                    with open(self.scene_metadata_path, "r", encoding="utf-8") as f:
                        self._scene_metadata = json.load(f)
                    self._scene_is_loaded = True
                    logger.info(
                        f"Loaded FAISS Scene index with {self._scene_index.ntotal} vectors"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load scene index: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def ensure_loaded(self) -> None:
        """Ensure the index is loaded."""
        if not self._is_loaded:
            if not self.load():
                raise RuntimeError("Index not available. Build or load an index first.")

    def search_by_text(
        self, query: str, k: int = 10, movie_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for images using a text query.

        Args:
            query: Text query describing the desired image
            k: Number of results to return
            movie_id: Optional filter to search within a specific movie

        Returns:
            List of SearchResult objects
        """
        self.ensure_loaded()

        # Encode query
        query_embedding = self.encoder.encode_text(query, normalize=True)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self._index.search(query_embedding, k * 2 if movie_id else k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]

            # Apply movie filter if specified
            if movie_id and meta.get("movie_id") != movie_id:
                continue

            results.append(
                SearchResult(
                    id=meta["id"],
                    path=meta["path"],
                    movie_id=meta.get("movie_id", "unknown"),
                    score=float(score),
                    metadata=meta,
                )
            )

            if len(results) >= k:
                break

        return results

    def search_scene_by_text(
        self, query: str, k: int = 5, movie_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for coherent scenes using a text query (HierRAG L1).
        Provides broader contextual understanding than single-frame matching.
        """
        self.ensure_loaded()
        if not self._scene_is_loaded or self._scene_index is None:
            logger.warning("Scene index not loaded. Falling back to frame search.")
            return self.search_by_text(query, k, movie_id)

        try:
            # Encode text query
            query_emb = self.encoder.encode_text(query)
            query_emb = query_emb.reshape(1, -1).astype(np.float32)

            # Search in faiss scene index
            D, indices_arr = self._scene_index.search(query_emb, k * 5)

            results = []
            seen_scenes = set()

            for i, idx in enumerate(indices_arr[0]):
                if idx < 0 or idx >= len(self._scene_metadata):
                    continue

                meta = self._scene_metadata[idx]
                scene_id = meta.get("id", "unknown")

                # Filter by movie
                if movie_id and meta.get("movie_id") != movie_id:
                    continue

                if scene_id in seen_scenes:
                    continue
                seen_scenes.add(scene_id)

                results.append(
                    SearchResult(
                        id=scene_id,
                        path=meta.get(
                            "keyframe_path", "aggregate"
                        ),  # fallback to one of the frames if exists
                        movie_id=meta.get("movie_id", "unknown"),
                        score=float(D[0][i]),
                        metadata=meta,
                    )
                )

                if len(results) >= k:
                    break

            return results
        except Exception as e:
            logger.error(f"Scene search error: {e}")
            return []

    def search_by_image(
        self,
        image: Union[str, "Image.Image"],
        k: int = 10,
        movie_id: Optional[str] = None,
        exclude_same: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar images using an image query.

        Args:
            image: Image path or PIL Image object
            k: Number of results to return
            movie_id: Optional filter to search within a specific movie
            exclude_same: Exclude exact matches (useful for finding duplicates)

        Returns:
            List of SearchResult objects
        """
        self.ensure_loaded()

        # Encode query image
        query_embedding = self.encoder.encode_image(image, normalize=True)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Search (get extra results in case we need to filter)
        search_k = k * 2 if movie_id or exclude_same else k
        scores, indices = self._index.search(query_embedding, search_k)

        # Build results
        results = []
        query_path = str(image) if isinstance(image, (str, Path)) else None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]

            # Exclude same image
            if exclude_same and query_path and meta["path"] == query_path:
                continue

            # Apply movie filter if specified
            if movie_id and meta.get("movie_id") != movie_id:
                continue

            results.append(
                SearchResult(
                    id=meta["id"],
                    path=meta["path"],
                    movie_id=meta.get("movie_id", "unknown"),
                    score=float(score),
                    metadata=meta,
                )
            )

            if len(results) >= k:
                break

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._is_loaded:
            return {"status": "not_loaded", "index_path": str(self.index_path)}

        return {
            "status": "loaded",
            "num_vectors": self._index.ntotal if self._index else 0,
            "embedding_dim": self.encoder.embedding_dim,
            "index_path": str(self.index_path),
            "metadata_entries": len(self._metadata),
        }

    #  Cross-encoder Re-ranking 

    def rerank_by_image(
        self,
        query_image: Union[str, "Image.Image"],
        candidates: List[SearchResult],
        preferred_movie_id: Optional[str] = None,
        movie_boost: float = 0.05,
    ) -> List[SearchResult]:
        """
        Cross-encoder re-ranking for image queries.

        Re-computes EXACT CLIP cosine similarity between the query image
        and each candidate's actual image file. This is more precise than
        FAISS approximate nearest neighbor search.

        Args:
            query_image: User's uploaded image
            candidates: FAISS search results to re-rank
            preferred_movie_id: Boost candidates from this movie
            movie_boost: Score boost for preferred movie matches

        Returns:
            Re-ranked list of SearchResult (highest score first)
        """
        if not candidates or len(candidates) <= 1:
            return candidates

        try:
            # Encode query image once
            query_emb = self.encoder.encode_image(query_image, normalize=True)
            query_emb = query_emb.reshape(1, -1).astype(np.float32)

            reranked = []
            for r in candidates:
                candidate_path = r.path or r.metadata.get("path", "")
                new_score = r.score  # Default to FAISS score

                if candidate_path and Path(candidate_path).exists():
                    try:
                        cand_emb = self.encoder.encode_image(
                            candidate_path, normalize=True
                        )
                        cand_emb = cand_emb.reshape(1, -1).astype(np.float32)
                        # Exact cosine similarity
                        new_score = float(np.dot(query_emb, cand_emb.T)[0, 0])
                    except Exception as e:
                        logger.warning(f"Reranking image fallback to FAISS score due to error: {e}")
                        pass  # Keep FAISS score on failure

                # Movie preference boost
                if preferred_movie_id and r.movie_id == preferred_movie_id:
                    new_score += movie_boost

                reranked.append(
                    SearchResult(
                        id=r.id,
                        path=r.path,
                        movie_id=r.movie_id,
                        score=new_score,
                        metadata=r.metadata,
                    )
                )

            # Sort by new score descending
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked

        except Exception as e:
            logger.warning(f"Rerank by image failed: {e}. Returning original order.")
            return candidates

    def rerank_by_text(
        self,
        query_text: str,
        candidates: List[SearchResult],
        preferred_movie_id: Optional[str] = None,
        movie_boost: float = 0.05,
    ) -> List[SearchResult]:
        """
        Cross-encoder re-ranking for text queries.

        Re-computes EXACT CLIP text-to-image cosine similarity by
        encoding each candidate's actual image against the text query.

        Args:
            query_text: User's text query
            candidates: FAISS search results to re-rank
            preferred_movie_id: Boost candidates from this movie
            movie_boost: Score boost for preferred movie matches

        Returns:
            Re-ranked list of SearchResult (highest score first)
        """
        if not candidates or len(candidates) <= 1:
            return candidates

        try:
            # Encode text query once
            text_emb = self.encoder.encode_text(query_text, normalize=True)
            text_emb = text_emb.reshape(1, -1).astype(np.float32)

            reranked = []
            for r in candidates:
                candidate_path = r.path or r.metadata.get("path", "")
                new_score = r.score  # Default to FAISS score

                if candidate_path and Path(candidate_path).exists():
                    try:
                        img_emb = self.encoder.encode_image(
                            candidate_path, normalize=True
                        )
                        img_emb = img_emb.reshape(1, -1).astype(np.float32)
                        new_score = float(np.dot(text_emb, img_emb.T)[0, 0])
                    except Exception as e:
                        logger.warning(f"Reranking fallback to FAISS score due to error: {e}")
                        pass # Keep FAISS score

                if preferred_movie_id and r.movie_id == preferred_movie_id:
                    new_score += movie_boost

                reranked.append(
                    SearchResult(
                        id=r.id,
                        path=r.path,
                        movie_id=r.movie_id,
                        score=new_score,
                        metadata=r.metadata,
                    )
                )

            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked

        except Exception as e:
            logger.warning(f"Rerank by text failed: {e}. Returning original order.")
            return candidates

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        coarse_k: int = 50,
        temporal_window: int = 3,
        alpha: float = 0.6,
        movie_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Multi-Grained Hybrid Search (Inspired by X-CLIP & Temporal Modeling).

        Stage 1 (Coarse): Fetch top `coarse_k` candidates using standard FAISS embedding.
        Stage 2 (Fine-Grained/Temporal): For each candidate, expand into a temporal scene
        (e.g. +/- `temporal_window` frames). Compute the similarity of the query to the
        entire temporal sequence (Sequence-Word matching).
        Final Score = alpha * Frame_Score + (1 - alpha) * Temporal_Score
        """
        self.ensure_loaded()
        if not self._is_loaded or self._index is None:
            logger.warning("Index not loaded. Returns empty.")
            return []

        # Encode query
        query_emb = self.encoder.encode_text(query).astype(np.float32).reshape(1, -1)

        # Stage 1: Coarse FAISS Search
        coarse_k_real = min(coarse_k, len(self._metadata))

        # Simple workaround for filtering by movie_id in FAISS without IDMap
        search_k = (
            coarse_k_real
            if not movie_id
            else min(coarse_k_real * 5, len(self._metadata))
        )
        distances, indices = self._index.search(query_emb, search_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            meta = self._metadata[idx]

            # Apply movie filter before temporal expansion to save compute
            if movie_id and meta.get("movie_id") != movie_id:
                continue

            frame_score = float(distances[0][i])

            # Stage 2: Temporal Scene Expansion
            start_idx = max(0, idx - temporal_window)
            end_idx = min(len(self._metadata) - 1, idx + temporal_window)

            temporal_scores = []
            for neighbor_idx in range(start_idx, end_idx + 1):
                try:
                    neighbor_vector = self._index.reconstruct(int(neighbor_idx))
                    sim = np.dot(query_emb[0], neighbor_vector)
                    temporal_scores.append(sim)
                except Exception:
                    pass

            # Sequence-Word score (using Mean aggregation approximation)
            temporal_score = (
                float(np.mean(temporal_scores)) if temporal_scores else frame_score
            )

            # Hybrid Fusion
            final_score = alpha * frame_score + (1.0 - alpha) * temporal_score

            results.append(
                SearchResult(
                    id=meta["id"],
                    path=meta["path"],
                    movie_id=meta["movie_id"],
                    score=final_score,
                    metadata={
                        **meta,
                        "hybrid_scores": {
                            "frame": frame_score,
                            "temporal": temporal_score,
                            "alpha": alpha,
                        },
                    },
                )
            )

            if len(results) >= coarse_k_real:
                break

        # Re-rank by final score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def hierarchical_search(
        self,
        query: str,
        k: int = 5,
        scene_k: int = 3,
        movie_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Hierarchical Scene→Frame drill-down search (SceneRAG architecture).

        Step 1: Search L1 scene index for top matching scenes.
        Step 2: For each matched scene, retrieve its L0 frames from the frame index.
        Step 3: Re-rank all candidate frames by their individual similarity.

        Args:
            query: Text query
            k: Number of frame-level results to return
            scene_k: Number of scenes to drill into
            movie_id: Optional movie filter

        Returns:
            List of SearchResult with frame-level detail + scene context
        """
        self.ensure_loaded()

        # If no scene index, fall back to hybrid search
        if not self._scene_is_loaded or self._scene_index is None:
            logger.info("No scene index → falling back to hybrid_search")
            return self.hybrid_search(query, k=k, movie_id=movie_id)

        # Step 1: Scene-level search (L1)
        scene_results = self.search_scene_by_text(query, k=scene_k, movie_id=movie_id)
        if not scene_results:
            return self.search_by_text(query, k=k, movie_id=movie_id)

        # Step 2: Collect shot IDs from matched scenes
        target_shot_ids = set()
        scene_context = {}  # shot_id → scene metadata
        for sr in scene_results:
            shot_ids = sr.metadata.get("shot_ids", [])
            for sid in shot_ids:
                target_shot_ids.add(sid)
                scene_context[sid] = {
                    "scene_id": sr.id,
                    "scene_score": sr.score,
                    "scene_timestamp": sr.metadata.get("timestamp", 0),
                    "scene_timestamp_end": sr.metadata.get("timestamp_end", 0),
                }

        # Step 3: Retrieve L0 frames belonging to those shots
        query_emb = self.encoder.encode_text(query, normalize=True)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        # Search the full frame index
        search_k = min(len(self._metadata), k * 20)
        distances, indices = self._index.search(query_emb, search_k)

        frame_results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]
            shot_id = meta.get("shot_id", "")

            # Only keep frames from matched scene's shots
            if shot_id not in target_shot_ids:
                continue

            if movie_id and meta.get("movie_id") != movie_id:
                continue

            # Enrich with scene context
            enriched_meta = {**meta}
            if shot_id in scene_context:
                enriched_meta["scene_context"] = scene_context[shot_id]

            frame_results.append(
                SearchResult(
                    id=meta["id"],
                    path=meta.get("path", ""),
                    movie_id=meta.get("movie_id", "unknown"),
                    score=float(distances[0][i]),
                    metadata=enriched_meta,
                )
            )

            if len(frame_results) >= k:
                break

        # If drill-down yielded too few results, supplement with direct search
        if len(frame_results) < k:
            direct = self.search_by_text(
                query, k=k - len(frame_results), movie_id=movie_id
            )
            seen_ids = {r.id for r in frame_results}
            for dr in direct:
                if dr.id not in seen_ids:
                    frame_results.append(dr)

        return frame_results[:k]

    def extract_video_clip(
        self,
        movie_id: str,
        frame_path: str,
        duration: int = 10,
        output_dir: str = "data/temp_clips",
    ) -> Optional[str]:
        """
        Extracts a chunk of raw MP4 video using ffmpeg based on a frame path.
        Estimates start time from timestamp or shot_id embedded in the filename.
        """
        try:
            raw_video_path = Path(f"data/raw_videos/{movie_id}.mp4")
            if not raw_video_path.exists():
                raw_video_path = Path(f"../data/raw_videos/{movie_id}.mp4")

            if not raw_video_path.exists():
                logger.debug(f"Raw video not found: {movie_id}.mp4")
                return None

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            import re

            # Try to parse timestamp from filepath (HH:MM:SS format)
            timestamp_match = re.search(
                r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)", str(frame_path)
            )
            if timestamp_match:
                start_time = timestamp_match.group(1)
            else:
                # Estimate from shot number (e.g. shot_0023 → ~23s at 24fps, rough)
                shot_match = re.search(
                    r"shot[_-]?(\d+)", str(frame_path), re.IGNORECASE
                )
                shot_num = int(shot_match.group(1)) if shot_match else 0
                # Approximate: each shot ~3-5s, use 3s per shot
                start_sec = shot_num * 3
                start_time = f"{start_sec // 3600:02d}:{(start_sec % 3600) // 60:02d}:{start_sec % 60:02d}"

            return self.extract_clip_at_time(
                movie_id, start_time, duration=duration, output_dir=output_dir
            )

        except Exception as e:
            logger.error(f"Failed to extract video clip from frame path: {e}")
            return None

    def extract_clip_at_time(
        self,
        movie_id: str,
        start_time: str,
        end_time: Optional[str] = None,
        duration: int = 15,
        output_dir: str = "data/temp_clips",
    ) -> Optional[str]:
        """
        Extracts a video clip using explicit HH:MM:SS timestamps.
        This is the primary method called from the temporal grounding flow.

        Args:
            movie_id: IMDb ID or movie identifier (e.g. 'tt0120338')
            start_time: Start time string in HH:MM:SS format
            end_time: End time string (optional, uses duration if not set)
            duration: Clip duration in seconds (used when end_time is None)
            output_dir: Directory for temp clips
        """
        try:
            raw_video_path = Path(f"data/raw_videos/{movie_id}.mp4")
            if not raw_video_path.exists():
                raw_video_path = Path(f"../data/raw_videos/{movie_id}.mp4")

            if not raw_video_path.exists():
                logger.debug(f"Raw video not found: {movie_id}.mp4")
                return None

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Calculate duration from start/end if both available
            if end_time:
                try:

                    def to_secs(t: str) -> float:
                        parts = list(map(float, t.split(":")))
                        if len(parts) == 3:
                            return parts[0] * 3600 + parts[1] * 60 + parts[2]
                        elif len(parts) == 2:
                            return parts[0] * 60 + parts[1]
                        return float(parts[0])

                    duration = max(3, int(to_secs(end_time) - to_secs(start_time)))
                except Exception:
                    pass

            safe_ts = start_time.replace(":", "-").replace(".", "-")
            out_file = out_dir / f"{movie_id}_{safe_ts}_{duration}s.mp4"
            if out_file.exists():
                return str(out_file)

            # ffmpeg: fast seek before input (-ss before -i) for speed, then limit duration
            import subprocess

            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                start_time,
                "-i",
                str(raw_video_path),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                str(out_file),
            ]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode()[:500]}")
                return None

            logger.info(
                f"Extracted clip: {out_file} ({duration}s from {start_time})"
            )
            return str(out_file)

        except FileNotFoundError:
            logger.error("ffmpeg not found. Install ffmpeg and add to PATH.")
            return None
        except Exception as e:
            logger.error(f"Failed to extract clip at time: {e}")
            return None


class VisualSearchEngine:
    """
    High-level search engine combining visual indexing with MovieNet data.
    """

    def __init__(self, indexer: VisualIndexer, movienet_loader: Optional[Any] = None):
        """
        Initialize the search engine.

        Args:
            indexer: VisualIndexer instance
            movienet_loader: Optional MovieNetLoader for enhanced metadata
        """
        self.indexer = indexer
        self.movienet_loader = movienet_loader

    def search(
        self,
        query: Union[str, "Image.Image"],
        k: int = 10,
        movie_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search using either text or image query.

        Args:
            query: Text string or PIL Image
            k: Number of results
            movie_id: Optional movie filter

        Returns:
            List of SearchResult objects
        """
        if isinstance(query, str):
            return self.indexer.search_by_text(query, k=k, movie_id=movie_id)
        else:
            return self.indexer.search_by_image(query, k=k, movie_id=movie_id)

    def identify_movie(
        self, image: Union[str, "Image.Image"], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify which movie(s) an image might be from.

        Args:
            image: Query image
            top_k: Number of movie candidates to return

        Returns:
            List of dicts with movie_id and confidence score
        """
        results = self.indexer.search_by_image(image, k=50)

        # Aggregate scores by movie
        movie_scores: Dict[str, float] = {}
        movie_counts: Dict[str, int] = {}

        for result in results:
            mid = result.movie_id
            if mid not in movie_scores:
                movie_scores[mid] = 0.0
                movie_counts[mid] = 0
            movie_scores[mid] += result.score
            movie_counts[mid] += 1

        # Average scores
        movie_results = [
            {
                "movie_id": mid,
                "avg_score": movie_scores[mid] / movie_counts[mid],
                "match_count": movie_counts[mid],
            }
            for mid in movie_scores
        ]

        # Sort by score
        movie_results.sort(key=lambda x: x["avg_score"], reverse=True)

        return movie_results[:top_k]
