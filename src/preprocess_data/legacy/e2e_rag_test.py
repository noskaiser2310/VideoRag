import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Study\School\project_ky4\src")))

from movierag.indexing.parallel_indexer import ParallelVisualIndexer
from movierag.indexing.knowledge_indexer import KnowledgeIndexer
from movierag.data.unified_loader import UnifiedLoader
from movierag.routing.query_router import QueryRouter
from movierag.generation.llm_generator import LLMGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(r"D:\Study\School\project_ky4")

logger.info("--- BOOTING MOVIERAG TESTING HARNESS ---")

# 1. Load Visual Index
vis_index_dir = str(PROJECT_ROOT / "data" / "unified_dataset")
visual_indexer = ParallelVisualIndexer(
    index_dir=vis_index_dir, index_name="movie_hybrid_index"
)
visual_indexer.load()

# 2. Load Knowledge Index
loader = UnifiedLoader(data_root=str(PROJECT_ROOT / "data"))
knowledge_indexer = KnowledgeIndexer(
    index_dir=str(PROJECT_ROOT / "data" / "indexes"), index_name="movierag_index"
)
knowledge_indexer.load()

# 3. Load Graph Index
from movierag.indexing.graph_indexer import GraphIndexer

graph_indexer = GraphIndexer(
    index_dir=str(PROJECT_ROOT / "data" / "unified_dataset"),
    index_name="movie_graph_index",
)
try:
    graph_indexer.load()
except Exception as e:
    logger.warning(f"Graph index not loaded: {e}")

# 4. Load Router & LLM
router = QueryRouter()
llm = LLMGenerator()


def run_test_query(query_text):
    print(f"\n=================================")
    print(f"QUERY: {query_text}")
    print(f"=================================")

    # Routing
    route = router.route_query(query_text)
    print(f"[ROUTER] Identified Route: {route.value}")

    # Visual Search
    v_results = visual_indexer.search_by_text(query_text, k=2)
    print("\n[VISUAL] Top 2 Matches:")
    for r in v_results:
        print(f"   -> {r.movie_id} (Score: {r.score:.4f}) / Path: {Path(r.path).name}")

    # Knowledge Search
    k_results = knowledge_indexer.search(query_text, k=2)
    print("\n[KNOWLEDGE] Top Matches:")
    for r in k_results:
        cat = r.metadata.get("category", "N/A")
        print(f"   -> {r.movie_id} (Score: {r.score:.4f}) / Type: {cat}")

    # Graph Search (VideoRAG Dual-Channel)
    try:
        from movierag.indexing.knowledge_indexer import TextSearchResult

        g_results = graph_indexer.search(query_text, k=2)
        print("\n[GRAPH KNOWLEDGE] Top Matches:")
        for res in g_results:
            print(
                f"   -> {res.get('movie_id')} (Score: {res.get('score', 0)}) / Chunk: {res.get('clip_id')}"
            )
            k_results.append(
                TextSearchResult(
                    movie_id=res.get("movie_id", "Unknown"),
                    clip_id=res.get("clip_id", "Unknown"),
                    text="[GRAPH EXTRACTED] " + res.get("text", ""),
                    score=float(res.get("score", 1.0)),
                    metadata={"category": "GraphRAG Grounding"},
                )
            )
    except Exception as e:
        print(f"[GRAPH ERROR] {e}")

    # LLM Generate
    try:
        ans = llm.generate_answer(
            query_text, k_results, v_results, history=[], route=route
        )
        print(f"\n[LLM RESPONSE]\n{ans}")
    except Exception as e:
        print(f"[LLM ERROR] {e}")


# Run tests
queries = [
    "Who directed the movie Inception?",
    "Why did the two men fight in the alleyway?",
    "Find a scene where a car blows up at night.",
]

for q in queries:
    run_test_query(q)

print("\n--- TESTS COMPLETE ---")
