"""
Graph Indexer for MovieRAG, implementing VideoRAG's Dual-Channel GraphRAG approach.

Extracts Entities and Relationships from transcripts using Gemini to build
a Semantic Knowledge Graph for high-level reasoning queries.
"""

import os
import json
import logging
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional

from movierag.generation.universal_client import UniversalLLMClient

# Provide an LLM driver flag
LLM_AVAILABLE = True

logger = logging.getLogger(__name__)


class GraphIndexer:
    """
    Indexes textual knowledge from movie transcripts into a Graph structure.
    Uses Gemini to extract Entities and Relationships, creating nodes and edges.
    """

    def __init__(
        self,
        index_dir: str,
        index_name: str = "movie_graph",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Graph Indexer.

        Args:
            index_dir: Directory to store/load index files
            index_name: Base name for index files
            api_key: Google Gemini API key
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_name = index_name
        self.graph_path = self.index_dir / f"{index_name}.graphml"

        # Priority to provided key, then environment
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if LLM_AVAILABLE and self.api_key:
            try:
                self.client = UniversalLLMClient()
            except Exception as e:
                logger.warning(
                    f"Could not initialize UniversalLLMClient for GraphIndexer: {e}"
                )
                self.client = None
        else:
            self.client = None
            logger.warning("No API key provided. Graph extraction requires an LLM.")

        self.graph = nx.Graph()
        self._is_loaded = False

    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract Graph Data from a text chunk."""
        if not self.client:
            return {"entities": [], "relationships": []}

        prompt = f"""
        Extract key entities (Characters, Locations, Objects) and their relationships from the following movie transcript.
        Return ONLY a JSON object with this exact structure:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "Character/Location/Object", "description": "Brief description"}}
            ],
            "relationships": [
                {{"source": "Entity 1", "target": "Entity 2", "description": "How they relate or interact"}}
            ]
        }}
        
        Transcript:
        {text[:4000]}
        """
        try:
            prompt += "\nMUST OUTPUT ONLY RAW JSON! NO MARKDOWN BLOCKS."
            response = self.client.models.generate_content(
                model="moonshotai/kimi-k2-instruct",
                contents=prompt,
            )
            text = response.text.replace("```json\n", "").replace("\n```", "").strip()
            return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to extract graph data: {e}")
            return {"entities": [], "relationships": []}

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the Knowledge Graph index from textual documents.
        """
        logger.info(f"Building Graph Knowledge Base from {len(documents)} documents...")
        self.graph.clear()

        # Provide a quick mock processing for speed if necessary,
        # but normally this iterates over all docs.
        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            if len(text.strip()) < 10:
                continue

            movie_id = doc.get("movie_id", f"movie_{i}")
            clip_id = doc.get("clip_id", f"clip_{i}")
            scene_label = doc.get("metadata", {}).get("scene_label", "")

            logger.debug(f"Extracting graph data for {movie_id} - {clip_id}")
            graph_data = self.extract_entities_and_relations(text)

            # 1. Add Scene Chunk Node
            if not self.graph.has_node(clip_id):
                self.graph.add_node(
                    clip_id,
                    type="SceneChunk",
                    text=text,
                    movie_id=movie_id,
                    scene_label=scene_label,
                )

            # 2. Add Entities
            for ent in graph_data.get("entities", []):
                ent_name = str(ent.get("name", "Unknown"))
                node_id = f"{movie_id}_{ent_name}".upper()
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        name=ent_name,
                        type=str(ent.get("type", "Unknown")),
                        description=str(ent.get("description", "")),
                        movie_id=movie_id,
                    )
                # Link entity to the chunk where it was mentioned
                self.graph.add_edge(node_id, clip_id, relation="MENTIONED_IN")

            # 3. Add Relationships
            for rel in graph_data.get("relationships", []):
                src_name = str(rel.get("source", "Unknown"))
                tgt_name = str(rel.get("target", "Unknown"))
                src_id = f"{movie_id}_{src_name}".upper()
                tgt_id = f"{movie_id}_{tgt_name}".upper()

                # Ensure nodes exist (fallback)
                if not self.graph.has_node(src_id):
                    self.graph.add_node(
                        src_id, name=src_name, type="Unknown", movie_id=movie_id
                    )
                if not self.graph.has_node(tgt_id):
                    self.graph.add_node(
                        tgt_id, name=tgt_name, type="Unknown", movie_id=movie_id
                    )

                self.graph.add_edge(
                    src_id, tgt_id, relation=str(rel.get("description", "Related"))
                )

        self._is_loaded = True
        self.save()
        logger.info(
            f"Graph index built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )

    def save(self):
        """Save networkx graph to disk."""
        nx.write_graphml(self.graph, str(self.graph_path))
        logger.info(f"Saved Graph index to {self.graph_path}")

    def load(self) -> bool:
        """Load networkx graph from disk."""
        if not self.graph_path.exists():
            logger.warning(f"Graph file not found at {self.graph_path}")
            return False
        self.graph = nx.read_graphml(str(self.graph_path))
        self._is_loaded = True
        logger.info(f"Loaded Graph index with {self.graph.number_of_nodes()} nodes")
        return True

    def ensure_loaded(self):
        if not self._is_loaded:
            if not self.load():
                raise RuntimeError("Graph index not found. Build index first.")

    def search(
        self, query: str, k: int = 5, movie_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Graph-based retrieval.
        1. Extract entities from query
        2. Match to graph nodes
        3. Traverse 1-hop to find heavily connected SceneChunks
        """
        self.ensure_loaded()
        if not self.client:
            logger.warning("No LLM client for query entity extraction.")
            return []

        # 1. Extract entities from the user's query
        prompt = f"Extract the key entities (Characters, Locations, Objects) from this query. Return ONLY a JSON list of strings (NO MARKDOWN). Query: {query}"
        try:
            resp = self.client.models.generate_content(
                model="moonshotai/kimi-k2-instruct",
                contents=prompt,
            )
            text = resp.text.replace("```json\n", "").replace("\n```", "").strip()
            query_entities = json.loads(text)
        except Exception as e:
            logger.error(f"Failed to extract entities from query: {e}")
            return []

        # 2. Find matching nodes in the graph
        found_nodes = []
        for ent in query_entities:
            ent_lower = ent.lower()
            for node, data in self.graph.nodes(data=True):
                if data.get("type") != "SceneChunk":
                    name = str(data.get("name", "")).lower()
                    if ent_lower in name or name in ent_lower:
                        if movie_id and data.get("movie_id") != movie_id:
                            continue
                        found_nodes.append(node)

        # 3. Traverse 1 hop to get SceneChunks
        chunk_scores = {}
        for node in found_nodes:
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor].get("type") == "SceneChunk":
                    # Weight by number of query entities connected to this chunk
                    chunk_scores[neighbor] = chunk_scores.get(neighbor, 0) + 1

        # Sort chunks by number of entity matches
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        results = []
        for chunk_id, score in sorted_chunks:
            data = self.graph.nodes[chunk_id]
            results.append(
                {
                    "clip_id": chunk_id,
                    "text": data.get("text", ""),
                    "movie_id": data.get("movie_id", ""),
                    "score": score,
                }
            )

        return results
