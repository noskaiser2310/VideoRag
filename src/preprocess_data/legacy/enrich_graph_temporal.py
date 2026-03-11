"""
️ Neo4j (NetworkX) Graph Enrichment with Temporal Chunks

Loads the existing movie knowledge graph and enriches it with TemporalChunk
nodes from `data/temporal_chunks/all_chunks.json`.

New nodes and edges added:
  (:TemporalChunk {chunk_id, start_time, end_time, description, dialogue_text, ...})
  (:TemporalChunk)-[:BELONGS_TO]->(:Movie)
  (:Character)-[:APPEARS_IN]->(:TemporalChunk)
  (:TemporalChunk)-[:FOLLOWED_BY]->(:TemporalChunk)   # sequential scene ordering

Usage:
    python scripts/enrich_graph_temporal.py
    python scripts/enrich_graph_temporal.py --movie tt0120338
"""

import json
import sys
import logging
import argparse
from pathlib import Path

import networkx as nx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("GraphEnrichment")

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = BASE_DIR / "temporal_chunks" / "all_chunks.json"
GRAPH_PATH = BASE_DIR / "unified_dataset" / "movie_graph_index.graphml"
OUTPUT_GRAPH_PATH = BASE_DIR / "unified_dataset" / "movie_graph_enriched.graphml"


def main():
    parser = argparse.ArgumentParser(
        description="Enrich Knowledge Graph with Temporal Chunks"
    )
    parser.add_argument("--movie", type=str, help="Process single movie only")
    parser.add_argument(
        "--graph", type=str, default=str(GRAPH_PATH), help="Path to existing .graphml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_GRAPH_PATH),
        help="Path to save enriched graph",
    )
    args = parser.parse_args()

    #  Load existing graph 
    graph_path = Path(args.graph)
    if graph_path.exists():
        logger.info(f"Loading existing graph from: {graph_path}")
        G = nx.read_graphml(str(graph_path))
        logger.info(
            f"  Existing nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}"
        )
    else:
        logger.info("No existing graph found. Creating new graph.")
        G = nx.Graph()

    #  Load temporal chunks 
    logger.info(f"Loading temporal chunks from: {CHUNKS_PATH}")
    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    logger.info(f"  Total chunks: {len(chunks)}")

    if args.movie:
        chunks = [c for c in chunks if c["movie_id"] == args.movie]
        logger.info(f"  Filtered to {len(chunks)} chunks for movie {args.movie}")

    #  Stats counters 
    stats = {
        "chunks_added": 0,
        "movie_nodes_added": 0,
        "character_nodes_added": 0,
        "belongs_to_edges": 0,
        "appears_in_edges": 0,
        "followed_by_edges": 0,
        "acted_by_edges": 0,
    }

    #  Group chunks by movie for FOLLOWED_BY edges 
    movie_chunks = {}
    for chunk in chunks:
        mid = chunk["movie_id"]
        if mid not in movie_chunks:
            movie_chunks[mid] = []
        movie_chunks[mid].append(chunk)

    # Sort each movie's chunks by start_seconds for temporal ordering
    for mid in movie_chunks:
        movie_chunks[mid].sort(key=lambda c: c.get("start_seconds", 0))

    #  Process each movie 
    for mid, m_chunks in movie_chunks.items():
        logger.info(f"\n  Processing {mid}: {len(m_chunks)} chunks")

        # 1. Ensure Movie node exists
        movie_node_id = f"MOVIE_{mid}"
        if not G.has_node(movie_node_id):
            title = m_chunks[0].get("title", mid) if m_chunks else mid
            genres = ", ".join(m_chunks[0].get("genres", [])) if m_chunks else ""
            G.add_node(
                movie_node_id,
                type="Movie",
                name=title,
                movie_id=mid,
                genres=genres,
            )
            stats["movie_nodes_added"] += 1

        prev_chunk_id = None

        for chunk in m_chunks:
            chunk_id = chunk["chunk_id"]

            # 2. Add TemporalChunk node
            if not G.has_node(chunk_id):
                G.add_node(
                    chunk_id,
                    type="TemporalChunk",
                    movie_id=mid,
                    start_time=chunk.get("start_time", ""),
                    end_time=chunk.get("end_time", ""),
                    start_seconds=str(chunk.get("start_seconds", 0)),
                    end_seconds=str(chunk.get("end_seconds", 0)),
                    duration_seconds=str(chunk.get("duration_seconds", 0)),
                    description=chunk.get("description", "")[:500],
                    situation=chunk.get("situation", ""),
                    scene_label=chunk.get("scene_label", ""),
                    dialogue_text=chunk.get("dialogue_text", "")[:500],
                    timestamp_source=chunk.get("timestamp_source", ""),
                )
                stats["chunks_added"] += 1

            # 3. BELONGS_TO edge: TemporalChunk → Movie
            if not G.has_edge(chunk_id, movie_node_id):
                G.add_edge(chunk_id, movie_node_id, relation="BELONGS_TO")
                stats["belongs_to_edges"] += 1

            # 4. APPEARS_IN edges: Character → TemporalChunk
            for char_name in chunk.get("characters", []):
                char_node_id = f"{mid}_{char_name}".upper()

                # Create character node if it doesn't exist in the graph
                if not G.has_node(char_node_id):
                    G.add_node(
                        char_node_id,
                        type="Character",
                        name=char_name,
                        movie_id=mid,
                    )
                    stats["character_nodes_added"] += 1

                if not G.has_edge(char_node_id, chunk_id):
                    G.add_edge(char_node_id, chunk_id, relation="APPEARS_IN")
                    stats["appears_in_edges"] += 1

            # 5. ACTED_BY edges: Character → Actor (from cast_in_scene)
            for cast_entry in chunk.get("cast_in_scene", []):
                actor_name = cast_entry.get("actor", "")
                char_name = cast_entry.get("character", "")
                if not actor_name or not char_name:
                    continue

                actor_node_id = f"{mid}_{actor_name}".upper()
                char_node_id = f"{mid}_{char_name}".upper()

                # Create actor node if needed
                if not G.has_node(actor_node_id):
                    G.add_node(
                        actor_node_id,
                        type="Actor",
                        name=actor_name,
                        movie_id=mid,
                    )

                # Create character node if needed
                if not G.has_node(char_node_id):
                    G.add_node(
                        char_node_id,
                        type="Character",
                        name=char_name,
                        movie_id=mid,
                    )

                if not G.has_edge(actor_node_id, char_node_id):
                    G.add_edge(actor_node_id, char_node_id, relation="PLAYS")
                    stats["acted_by_edges"] += 1

            # 6. FOLLOWED_BY edge: TemporalChunk → TemporalChunk (sequential)
            if prev_chunk_id and not G.has_edge(prev_chunk_id, chunk_id):
                G.add_edge(prev_chunk_id, chunk_id, relation="FOLLOWED_BY")
                stats["followed_by_edges"] += 1

            prev_chunk_id = chunk_id

    #  Save enriched graph 
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))

    # Also overwrite the original graph so the system picks up the enriched version
    nx.write_graphml(G, str(graph_path))

    #  Summary 
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ️ GRAPH ENRICHMENT COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(
        f"  Final graph:  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    logger.info(f"  New TemporalChunk nodes:   {stats['chunks_added']}")
    logger.info(f"  New Movie nodes:           {stats['movie_nodes_added']}")
    logger.info(f"  New Character nodes:       {stats['character_nodes_added']}")
    logger.info(f"  BELONGS_TO edges:          {stats['belongs_to_edges']}")
    logger.info(f"  APPEARS_IN edges:          {stats['appears_in_edges']}")
    logger.info(f"  FOLLOWED_BY edges:         {stats['followed_by_edges']}")
    logger.info(f"  PLAYS (Actor→Character):   {stats['acted_by_edges']}")
    logger.info(f"\n  Saved to:   {output_path}")
    logger.info(f"  Also saved: {graph_path}")

    # Count node types
    type_counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"\n  Node types: {json.dumps(type_counts, indent=2)}")


if __name__ == "__main__":
    main()
