import os
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from movienet_loader import MovieNetLoader
import logging

try:
    import faiss
except ImportError:
    # If FAISS is not installed/working, we can use a simple flat index or warn
    print("FAISS not found. Will use simple matrix multiplication for demo.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualIndexer:
    def __init__(self, index_path="visual_index.faiss"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load CLIP model (using openai/clip-vit-base-patch32 for balance)
        logger.info("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.index_path = index_path
        self.index = None
        self.image_paths = []  # Keep track of which ID maps to which path

    def encode_images(self, image_paths):
        """
        Encode a list of image paths into vectors.
        """
        vectors = []
        batch_size = 32

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            valid_paths = []

            for p in batch_paths:
                try:
                    images.append(Image.open(p))
                    valid_paths.append(p)
                except Exception as e:
                    logger.warning(f"Could not read {p}: {e}")

            if not images:
                continue

            inputs = self.processor(
                images=images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            vectors.append(image_features.cpu().numpy())

        if vectors:
            return np.vstack(vectors), valid_paths
        return np.array([]), []

    def build_index(self, loader: MovieNetLoader):
        """
        Build the FAISS index from the loader's data.
        """
        # 1. Get all shots
        data = loader.load_sample_data()
        all_shots = [s["path"] for s in data["shots"]]

        if not all_shots:
            logger.warning("No shots found to index.")
            return

        logger.info(f"Encoding {len(all_shots)} images...")
        embeddings, valid_paths = self.encode_images(all_shots)

        self.image_paths = valid_paths

        # 2. Key FAISS Index
        d = embeddings.shape[1]  # Dimension (e.g., 512)
        self.index = faiss.IndexFlatIP(
            d
        )  # Inner Product (Cosine Similarity because normalized)
        self.index.add(embeddings)

        logger.info(f"Built index with {self.index.ntotal} vectors.")

        # Save index and mapping
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".map", "w") as f:
            json.dump(self.image_paths, f)

    def search(self, query_text=None, query_image_path=None, k=5):
        """
        Search the index by text or image.
        """
        if self.index is None:
            # Try load
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.index_path + ".map", "r") as f:
                    self.image_paths = json.load(f)
            else:
                raise ValueError("Index not built or loaded.")

        query_vector = None

        if query_text:
            inputs = self.processor(
                text=[query_text], return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )
            query_vector = text_features.cpu().numpy()

        elif query_image_path:
            image = Image.open(query_image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            query_vector = image_features.cpu().numpy()

        else:
            return []

        # Search
        D, I = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(I[0]):
            if idx != -1:
                results.append({"path": self.image_paths[idx], "score": float(D[0][i])})

        return results


if __name__ == "__main__":
    import json

    # Setup paths
    base_path = r"d:\Study\School\project_ky4\data\movienet_tools\tests\data"
    loader = MovieNetLoader(base_path)

    indexer = VisualIndexer()

    # 1. Build Index (One time)
    logger.info("Building Index...")
    indexer.build_index(loader)

    # 2. Demo Search
    query = "a person standing"
    logger.info(f"Searching for: '{query}'")
    results = indexer.search(query_text=query)

    print("\nSearch Results:")
    for res in results:
        print(f"- [Score: {res['score']:.4f}] {res['path']}")
