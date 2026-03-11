"""
MovieRAG Main Pipeline
======================
Production-ready entry point for the MovieRAG system.

Usage:
    # Build index from data
    python -m movierag.main build --data-dir path/to/movienet/data

    # Run interactive demo
    python -m movierag.main demo

    # Run verification tests
    python -m movierag.main verify
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Load environment variables — search several candidate locations
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    _candidates = [
        _here.parent.parent / ".env",  # src/.env  (user's actual location)
        _here.parent.parent.parent / ".env",  # project_ky4/.env
        Path(".env"),  # CWD/.env
    ]
    for _env_path in _candidates:
        if _env_path.exists():
            load_dotenv(_env_path, override=True)
            import logging as _l

            _l.getLogger(__name__).info(f"Loaded .env from {_env_path}")
            break
except ImportError:
    pass


# Detect project root (contains src/ and data/)
def _find_project_root() -> Path:
    """Find project root by looking for src/ directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "src").exists() and (current / "data").exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "movie_data_subset_20")
DEFAULT_INDEX_DIR = str(PROJECT_ROOT / "data" / "indexes")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("movierag")


def build_index(
    data_dir: str,
    index_dir: str = "data/indexes",
    index_name: str = "movierag_index",
    use_sample: bool = False,
) -> bool:
    """
    Build the knowledge search index from all available data sources.

    Args:
        data_dir: Path to movie data directory (movie_data_subset_20)
        index_dir: Directory to store the index
        index_name: Name for the index files
        use_sample: Whether to use sample data structure

    Returns:
        True if successful, False otherwise
    """
    from movierag.config import get_config
    from movierag.data.unified_loader import UnifiedLoader
    from movierag.data.subtitle_loader import SubtitleLoader
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer

    cfg = get_config()

    logger.info(f"Building index from: {data_dir}")
    logger.info(f"Index will be saved to: {index_dir}/{index_name}")

    try:
        # Initialize loaders
        unified_loader = UnifiedLoader(data_root=str(cfg.paths.data_dir))
        subtitle_loader = SubtitleLoader(subtitle_dir=str(cfg.paths.subtitle_dir))

        movie_ids = unified_loader.get_all_movie_ids()
        logger.info(f"Found {len(movie_ids)} movies from unified dataset")

        subtitle_movies = subtitle_loader.get_available_movies()
        logger.info(f"Found {len(subtitle_movies)} movies with subtitles")

        # Build knowledge index from all loaders
        indexer = KnowledgeIndexer(index_dir=index_dir, index_name=index_name)
        indexer.build_from_loaders(
            unified_loader=unified_loader,
            subtitle_loader=subtitle_loader,
        )

        logger.info(f"[OK] Knowledge index built with {indexer.num_documents} vectors")
        return True

    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_demo(
    data_dir: str = DEFAULT_DATA_DIR,
    index_dir: str = DEFAULT_INDEX_DIR,
    index_name: str = "movierag_index",
    port: int = 7860,
) -> None:
    """
    Run the interactive Gradio demo.

    Args:
        data_dir: Path to MovieNet data
        index_dir: Directory containing the index
        index_name: Name of the index
        port: Port for the web server
    """
    from movierag.config import get_config
    from movierag.data.unified_loader import UnifiedLoader
    from movierag.data.subtitle_loader import SubtitleLoader
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer

    try:
        import gradio as gr  # noqa: F401 - Used via create_demo
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio")
        return

    cfg = get_config()

    logger.info("Starting MovieRAG System App...")

    # Initialize Visual Search
    from movierag.indexing.parallel_indexer import ParallelVisualIndexer

    vis_index_dir = str(cfg.paths.data_dir / "unified_dataset")

    logger.info("Loading Visual FAISS Index...")
    visual_indexer = ParallelVisualIndexer(
        index_dir=vis_index_dir, index_name="movie_hybrid_index"
    )
    try:
        visual_indexer.load()
    except Exception as e:
        logger.warning(f"Visual index not loaded: {e}")

    # Initialize Knowledge Search
    knowledge_indexer = KnowledgeIndexer(index_dir=index_dir, index_name=index_name)

    # Build or load knowledge index
    if not knowledge_indexer.index_path.exists():
        logger.info("Building knowledge FAISS index for all processed movies...")
        unified_loader = UnifiedLoader(data_root=str(cfg.paths.data_dir))
        subtitle_loader = SubtitleLoader(subtitle_dir=str(cfg.paths.subtitle_dir))
        knowledge_indexer.build_from_loaders(
            unified_loader=unified_loader,
            subtitle_loader=subtitle_loader,
        )
    else:
        logger.info("Loading existing knowledge FAISS index...")
        knowledge_indexer.load()

    # Import and run integrated app
    from movierag.app import create_integrated_app
    from movierag.generation.llm_generator import LLMGenerator

    # Initialize LLM Generator
    llm_generator = LLMGenerator()

    # Initialize Dialogue Indexer
    from movierag.indexing.dialogue_indexer import DialogueIndexer

    dialogue_indexer = DialogueIndexer()

    # Initialize Agentic Pipeline
    from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline

    pipeline = AgenticVideoRAGPipeline(
        visual_indexer=visual_indexer,
        knowledge_indexer=knowledge_indexer,
        dialogue_indexer=dialogue_indexer,
        llm_generator=llm_generator,
        model_id="gemma-3-27b-it",
    )

    app = create_integrated_app(pipeline=pipeline)
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        allowed_paths=[
            str(cfg.paths.project_root / "data"),
            str(cfg.paths.movie_subset_dir),
        ],
    )


def run_verify(data_dir: str = DEFAULT_DATA_DIR) -> bool:
    """
    Run verification tests on the pipeline.

    Args:
        data_dir: Path to test data

    Returns:
        True if all tests pass, False otherwise
    """
    from movierag.data.unified_loader import UnifiedLoader
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer

    logger.info("Running verification tests...")
    tests_passed = 0
    tests_total = 0

    # Test 1: Data Loading
    tests_total += 1
    try:
        loader = UnifiedLoader(data_root=str(PROJECT_ROOT / "data"))
        movie_ids = loader.get_all_movie_ids()
        documents = list(loader.get_all_textual_documents())

        if movie_ids and documents:
            logger.info(
                f"[OK] Data Loading: {len(movie_ids)} movies, {len(documents)} documents"
            )
            tests_passed += 1
        else:
            logger.error("[FAIL] Data Loading: No data found")
    except Exception as e:
        logger.error(f"[FAIL] Data Loading: {e}")

    # Test 2: Indexing
    tests_total += 1
    try:
        indexer = KnowledgeIndexer(
            index_dir=str(PROJECT_ROOT / "data" / "indexes_verify"),
            index_name="verify_test",
        )
        indexer.build_index(documents[:5])  # Only 5 for speed

        if indexer._index and indexer._index.ntotal == 5:
            logger.info("[OK] Indexing: Built index with 5 vectors")
            tests_passed += 1
        else:
            logger.error("[FAIL] Indexing: Failed to build index")
    except Exception as e:
        logger.error(f"[FAIL] Indexing: {e}")

    # Test 3: Search
    tests_total += 1
    try:
        results = indexer.search("What happens in the movie?", k=1)
        if results:
            logger.info(f"[OK] Search: Found {len(results)} results")
            tests_passed += 1
        else:
            logger.error("[FAIL] Search: No results")
    except Exception as e:
        logger.error(f"[FAIL] Search: {e}")

    # Summary
    logger.info(f"\n{'=' * 40}")
    logger.info(f"VERIFICATION: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        logger.info("ALL TESTS PASSED - System is ready!")
        return True
    else:
        logger.warning("Some tests failed - Please check the errors above")
        return False


def run_evaluation(
    data_dir: str = DEFAULT_DATA_DIR, dataset_file: str = "data/eval_queries.json"
) -> bool:
    """Run the LLM-as-a-judge automated evaluation framework."""
    from movierag.indexing.visual_indexer import VisualIndexer
    from movierag.indexing.knowledge_indexer import KnowledgeIndexer
    from movierag.indexing.dialogue_indexer import DialogueIndexer
    from movierag.generation.llm_generator import LLMGenerator
    from movierag.pipeline.agentic_pipeline import AgenticVideoRAGPipeline
    from movierag.evaluation.eval_framework import MovieRAGEvaluator
    from groq import Groq
    import os

    logger.info("Starting Evaluation Framework...")

    visual_indexer = VisualIndexer(index_dir=str(PROJECT_ROOT / "data" / "indexes"))
    visual_indexer.load()

    knowledge_indexer = KnowledgeIndexer(
        index_dir=str(PROJECT_ROOT / "data" / "indexes")
    )
    knowledge_indexer.load()

    dialogue_indexer = DialogueIndexer()
    llm_generator = LLMGenerator()

    pipeline = AgenticVideoRAGPipeline(
        visual_indexer=visual_indexer,
        knowledge_indexer=knowledge_indexer,
        dialogue_indexer=dialogue_indexer,
        llm_generator=llm_generator,
        model_id="gemma-3-27b-it",
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("Vui lòng set biến môi trường GROQ_API_KEY để dùng LLM Judge.")
        return False

    groq_client = Groq(api_key=api_key)
    eval_file_path = str(PROJECT_ROOT / dataset_file)

    evaluator = MovieRAGEvaluator(
        pipeline=pipeline, llm_client=groq_client, eval_file_path=eval_file_path
    )
    evaluator.run_eval()
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MovieRAG - Visual Search with Timestamp Retrieval"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the visual search index")
    build_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to MovieNet data directory",
    )
    build_parser.add_argument(
        "--index-dir",
        "-i",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="Directory to store the index",
    )
    build_parser.add_argument(
        "--index-name",
        "-n",
        type=str,
        default="movierag_index",
        help="Name for the index files",
    )
    build_parser.add_argument(
        "--sample", action="store_true", help="Use sample data structure"
    )

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to MovieNet data",
    )
    demo_parser.add_argument(
        "--port", "-p", type=int, default=7860, help="Web server port"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Run verification tests")
    verify_parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to test data",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run LLM automated evaluation")
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_queries.json",
        help="Path to the JSON evaluation dataset",
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a new raw movie into the RAG system"
    )
    ingest_parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to the raw video file (.mp4, .mkv)",
    )
    ingest_parser.add_argument(
        "--id",
        "-i",
        type=str,
        required=True,
        help="IMDb ID or unique identifier for the movie (e.g. tt0120338)",
    )
    ingest_parser.add_argument(
        "--srt",
        "-s",
        type=str,
        required=False,
        help="Path to the corresponding .srt subtitle file (optional)",
    )

    args = parser.parse_args()

    if args.command == "build":
        success = build_index(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            index_name=args.index_name,
            use_sample=args.sample,
        )
        sys.exit(0 if success else 1)

    elif args.command == "demo":
        run_demo(data_dir=args.data_dir, port=args.port)

    elif args.command == "verify":
        success = run_verify(data_dir=args.data_dir)
        sys.exit(0 if success else 1)

    elif args.command == "eval":
        success = run_evaluation(dataset_file=args.dataset)
        sys.exit(0 if success else 1)

    elif args.command == "ingest":
        from preprocess_data.ingest_movie import MovieIngester

        ingester = MovieIngester(args.video, args.id, args.srt)
        logger.info(f"Starting End-to-End Ingestion for {args.id}...")

        # Run steps
        success = ingester.extract_frames()
        if success:
            raw_dialogues = ingester.parse_srt()
            raw_chunks = ingester.chunk_dialogues(raw_dialogues)
            enriched = ingester.enrich_and_save_chunks(raw_chunks)
            ingester.push_to_faiss(enriched)
            logger.info("Ingestion completed successfully.")
            sys.exit(0)
        else:
            logger.error("Ingestion failed during frame extraction.")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
