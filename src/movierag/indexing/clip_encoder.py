"""
CLIP Encoder for visual and text embeddings.

Uses OpenAI's CLIP model for creating embeddings that can be used
for cross-modal similarity search.
"""

import logging
from typing import List, Optional, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    CLIP-based encoder for images and text.

    Provides unified interface for encoding both modalities into a shared
    embedding space for similarity search.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
        local_files_only: bool = False,
    ):
        """
        Initialize the CLIP encoder.

        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            batch_size: Batch size for encoding
            local_files_only: If True, only use locally cached HuggingFace models. Good for parallel indexing.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.local_files_only = local_files_only

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing CLIP encoder on {self.device}...")

        # Lazy loading - model will be loaded on first use
        self._model = None
        self._processor = None

    def _ensure_model_loaded(self):
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        from transformers import CLIPProcessor, CLIPModel

        logger.info(
            f"Loading CLIP model: {self.model_name} (local_only={self.local_files_only})"
        )
        self._model = CLIPModel.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        self._model = self._model.to(self.device)
        self._model.eval()

        self._processor = CLIPProcessor.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        logger.info("CLIP model loaded successfully")

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        self._ensure_model_loaded()
        return self._model.config.projection_dim

    def encode_images(
        self,
        images: Union[List[str], List[Image.Image]],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode images into embeddings.

        Args:
            images: List of image paths or PIL Image objects
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (N, embedding_dim)
        """
        self._ensure_model_loaded()

        all_embeddings = []

        # Create batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]

            # Load images if paths are provided
            pil_images = []
            for img in batch:
                if isinstance(img, str) or isinstance(img, Path):
                    try:
                        pil_images.append(Image.open(img).convert("RGB"))
                    except Exception as e:
                        logger.warning(f"Failed to load image {img}: {e}")
                        continue
                else:
                    pil_images.append(img)

            if not pil_images:
                continue

            # Process and encode
            inputs = self._processor(
                images=pil_images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                embeddings = self._model.get_image_features(**inputs)

            if hasattr(embeddings, "image_embeds"):
                embeddings = embeddings.image_embeds
            elif hasattr(embeddings, "pooler_output"):
                embeddings = embeddings.pooler_output
            elif not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings[0]

            if normalize:
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

            if show_progress:
                progress = min(i + self.batch_size, len(images))
                logger.info(f"Encoded {progress}/{len(images)} images")

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    def encode_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize embeddings

        Returns:
            numpy array of shape (N, embedding_dim)
        """
        self._ensure_model_loaded()

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP max length
            ).to(self.device)

            with torch.no_grad():
                embeddings = self._model.get_text_features(**inputs)

            if hasattr(embeddings, "text_embeds"):
                embeddings = embeddings.text_embeds
            elif hasattr(embeddings, "pooler_output"):
                embeddings = embeddings.pooler_output
            elif not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings[0]

            if normalize:
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text string.

        Args:
            text: Text string to encode
            normalize: Whether to L2-normalize embedding

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.encode_texts([text], normalize=normalize)
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def encode_image(
        self, image: Union[str, Image.Image], normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a single image.

        Args:
            image: Image path or PIL Image object
            normalize: Whether to L2-normalize embedding

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.encode_images(
            [image], normalize=normalize, show_progress=False
        )
        return embeddings[0] if len(embeddings) > 0 else np.array([])

    def compute_similarity(
        self, query_embedding: np.ndarray, target_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between query and target embeddings.

        Args:
            query_embedding: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            target_embeddings: Target embeddings of shape (N, embedding_dim)

        Returns:
            Similarity scores of shape (N,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity (dot product for normalized vectors)
        similarities = np.dot(target_embeddings, query_embedding.T).flatten()

        return similarities
