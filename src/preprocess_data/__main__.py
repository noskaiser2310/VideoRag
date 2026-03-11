"""
CLI Entry Point for preprocess_data

Usage:
    python -m preprocess_data extract  [--movie ID] [--force]    # Extract keyframes
    python -m preprocess_data build    [--movie ID]              # Build temporal chunks
    python -m preprocess_data index    [--movie ID]              # Build FAISS + graph
    python -m preprocess_data all      [--movie ID]              # Full pipeline
    python -m preprocess_data ingest   <video_path> [--id ID]    # NEW VIDEO ingest
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("preprocess_data")


def cmd_extract(args):
    """Extract keyframes at scene boundaries."""
    from .video.keyframe_extractor import KeyframeExtractor

    ext = KeyframeExtractor()
    if args.movie:
        ext.process_movie(args.movie, force=args.force)
    else:
        ext.process_all(force=args.force)


def cmd_build(args):
    """Build 5-layer temporal chunks."""
    from .temporal.chunk_builder import ChunkBuilder

    builder = ChunkBuilder()
    if args.movie:
        builder.build_all(movie_ids=[args.movie])
    else:
        builder.build_all()


def cmd_index(args):
    """Build FAISS index + enrich graph."""
    from .indexing.faiss_builder import FaissBuilder
    from .indexing.graph_builder import GraphBuilder

    logger.info("Step 1: Building FAISS index...")
    faiss = FaissBuilder()
    faiss.build()

    logger.info("Step 2: Enriching knowledge graph...")
    graph = GraphBuilder()
    graph.enrich(movie_id=args.movie if args.movie else None)


def cmd_all(args):
    """Run full pipeline: extract → build → index."""
    logger.info(" Running full preprocessing pipeline...")
    cmd_extract(args)
    cmd_build(args)
    cmd_index(args)
    logger.info(" Full pipeline complete!")


def cmd_ingest(args):
    """
     INGEST A NEW VIDEO — Full Automated Pipeline

    8-step automated pipeline for a completely new video:
      1. Copy video to raw_videos/
      2. Crawl metadata (IMDB/TMDB)
      3. Auto-annotate (shotdetect → scene boundaries)
      4. Generate subtitles (STT → SRT)
      5. Extract precision keyframes (3 per scene)
      6. Analyze scenes (VLM + LLM → MovieGraphs-equivalent)
      7. Build temporal chunks (5-layer merge)
      8. Index into FAISS + enrich knowledge graph

    Usage:
      python -m preprocess_data ingest path/to/movie.mp4 --id tt9999999
      python -m preprocess_data ingest path/to/movie.mp4 --id my_movie --srt subs.srt
    """
    from .config import PreprocessConfig as Cfg

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f" Video file not found: {video_path}")
        sys.exit(1)

    movie_id = args.id or video_path.stem
    logger.info(f"\n{'=' * 60}")
    logger.info(f"   INGESTING NEW VIDEO: {video_path.name}")
    logger.info(f"  Movie ID: {movie_id}")
    logger.info(f"{'=' * 60}")

    #  Step 1/8: Copy video to raw_videos/ 
    Cfg.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    target_video = Cfg.RAW_VIDEOS_DIR / f"{movie_id}{video_path.suffix}"
    if not target_video.exists():
        logger.info(f"\n[1/8] Copying video to {target_video}...")
        shutil.copy2(str(video_path), str(target_video))
    else:
        logger.info(f"\n[1/8] Video already at {target_video}")

    #  Step 2/8: Crawl metadata (IMDB/TMDB) 
    logger.info(f"\n[2/8] Fetching metadata...")
    meta = {}
    try:
        from .dataset.metadata_crawler import MetadataCrawler

        crawler = MetadataCrawler()
        meta = crawler.crawl(movie_id)
    except Exception as e:
        logger.warning(f"  Metadata crawl failed: {e} (continuing without)")
        meta = {"imdb_id": movie_id, "title": movie_id}

    #  Step 3/8: Auto-annotate (shotdetect → scene boundaries) 
    ann_path = Cfg.ANNOTATION_DIR / f"{movie_id}.json"
    if not ann_path.exists():
        logger.info(f"\n[3/8] Auto-annotating (shot detection)...")
        try:
            from .video.auto_annotator import AutoAnnotator

            annotator = AutoAnnotator()
            annotation = annotator.annotate(movie_id, target_video)
            if not annotation:
                logger.warning("  Auto-annotation failed, using fixed intervals")
        except Exception as e:
            logger.warning(f"  Auto-annotation error: {e} (will use fixed intervals)")
    else:
        logger.info(f"\n[3/8] Annotation already exists: {ann_path}")

    #  Step 4/8: Generate subtitles (STT) 
    srt_path = Cfg.SUBTITLE_DIR / f"{movie_id}.srt"
    if args.srt:
        # User provided SRT file
        srt_source = Path(args.srt)
        if srt_source.exists():
            Cfg.SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(srt_source), str(srt_path))
            logger.info(f"\n[4/8] Copied user-provided subtitles to {srt_path}")
    elif not srt_path.exists():
        logger.info(f"\n[4/8] Generating subtitles (STT)...")
        try:
            from .video.stt_generator import STTGenerator

            stt = STTGenerator()
            stt_result = stt.generate(movie_id, target_video)
            if stt_result:
                logger.info(f"  STT complete: {stt_result}")
            else:
                logger.warning("  STT generation failed (continuing without subtitles)")
        except Exception as e:
            logger.warning(f"  STT error: {e} (continuing without subtitles)")
    else:
        logger.info(f"\n[4/8] Subtitles already exist: {srt_path}")

    #  Step 5/8: Extract precision keyframes 
    logger.info(f"\n[5/8] Extracting precision keyframes...")
    from .video.keyframe_extractor import KeyframeExtractor

    extractor = KeyframeExtractor()

    # Check if annotation exists now (may have been auto-generated in step 3)
    ann_path = Cfg.ANNOTATION_DIR / f"{movie_id}.json"
    if ann_path.exists():
        logger.info(f"  Found annotation → using scene boundaries")
        result = extractor.process_movie(movie_id, force=True)
    else:
        logger.info(f"  No annotation → extracting at fixed 5s intervals")
        result = _extract_fixed_interval(movie_id, target_video, extractor)

    if result.get("keyframes", 0) == 0:
        logger.error(" No keyframes extracted! Aborting.")
        sys.exit(1)

    #  Step 6/8: Analyze scenes (VLM + LLM) 
    logger.info(f"\n[6/8] Analyzing scenes (VLM + LLM)...")
    try:
        from .video.scene_analyzer import SceneAnalyzer

        analyzer = SceneAnalyzer()
        clips = analyzer.analyze_movie(movie_id, meta=meta)
        if clips:
            logger.info(f"  Scene analysis: {len(clips)} clips generated")
        else:
            logger.warning("  Scene analysis produced no clips (continuing)")
    except Exception as e:
        logger.warning(f"  Scene analysis error: {e} (continuing without)")

    #  Step 7/8: Build temporal chunks 
    logger.info(f"\n[7/8] Building temporal chunks...")
    from .temporal.chunk_builder import ChunkBuilder

    builder = ChunkBuilder()
    chunks = builder.build_for_movie(movie_id)

    if chunks:
        import json

        Cfg.TEMPORAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        out = Cfg.TEMPORAL_CHUNKS_DIR / f"{movie_id}_chunks.json"
        out.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"  Saved {len(chunks)} chunks → {out}")

        # Also update all_chunks.json
        _merge_into_all_chunks(movie_id, chunks)
    else:
        logger.warning("  No chunks built")

    #  Step 8/8: Index into FAISS + enrich graph 
    logger.info(f"\n[8/8] Indexing into FAISS + enriching graph...")
    from .indexing.faiss_builder import FaissBuilder
    from .indexing.graph_builder import GraphBuilder

    faiss_builder = FaissBuilder()
    faiss_builder.build()

    graph_builder = GraphBuilder()
    graph_builder.enrich(movie_id=movie_id)

    #  Summary 
    logger.info(f"\n{'=' * 60}")
    logger.info(f"   INGEST COMPLETE: {movie_id}")
    logger.info(f"  Title: {meta.get('title', movie_id)}")
    logger.info(f"  Keyframes: {result.get('keyframes', 0)}")
    logger.info(f"  Chunks: {len(chunks) if chunks else 0}")
    logger.info(f"  Steps completed:")
    logger.info(f"     Video copied")
    logger.info(
        f"     Metadata: {'crawled' if meta.get('auto_generated') else 'loaded'}"
    )
    logger.info(
        f"     Annotation: {'auto-generated' if ann_path.exists() else 'fixed intervals'}"
    )
    logger.info(
        f"     Subtitles: {'available' if srt_path.exists() else 'not available'}"
    )
    logger.info(f"     Keyframes extracted")
    logger.info(f"     FAISS indexed + graph enriched")
    logger.info(f"  Ready for queries in MovieRAG!")
    logger.info(f"{'=' * 60}")


def _extract_fixed_interval(movie_id: str, video_path: Path, extractor):
    """
    For new videos without annotation: extract keyframes at fixed intervals
    and create a keyframe_index.json with computed timestamps.
    """
    import json
    import subprocess
    import time

    info = extractor.get_video_info(video_path)
    fps = info["fps"]
    duration = info["duration"]
    interval = 5.0  # Extract every 5 seconds for new videos

    from .config import PreprocessConfig as Cfg

    out_dir = Cfg.SHOT_KEYF_DIR / movie_id
    out_dir.mkdir(parents=True, exist_ok=True)

    keyframe_index = []
    extracted = 0
    scene_idx = 0
    t0 = time.time()

    ts = 0.0
    while ts < duration:
        for img_idx, offset in enumerate([0.0, interval / 2, interval - 0.1]):
            actual_ts = min(ts + offset, duration - 0.1)
            fname = f"shot_{scene_idx:04d}_img_{img_idx}.jpg"
            out_path = out_dir / fname

            if extractor.extract_frame_at_time(video_path, actual_ts, out_path):
                extracted += 1
                keyframe_index.append(
                    {
                        "filename": fname,
                        "scene_idx": scene_idx,
                        "scene_id": f"auto_scene_{scene_idx}",
                        "img_idx": img_idx,
                        "timestamp_sec": round(actual_ts, 3),
                        "timestamp_fmt": _fmt(actual_ts),
                        "scene_start_sec": round(ts, 3),
                        "scene_end_sec": round(min(ts + interval, duration), 3),
                        "path": str(out_path),
                    }
                )

        scene_idx += 1
        ts += interval

    # Save index
    index_data = {
        "movie_id": movie_id,
        "video_fps": fps,
        "video_duration": duration,
        "total_scenes": scene_idx,
        "total_keyframes": extracted,
        "extraction_method": "fixed_interval",
        "interval_seconds": interval,
        "keyframes": keyframe_index,
    }
    (out_dir / "keyframe_index.json").write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    logger.info(
        f"  Extracted {extracted} keyframes in {elapsed:.1f}s (every {interval}s)"
    )
    return {"movie_id": movie_id, "status": "ok", "keyframes": extracted}


def _merge_into_all_chunks(movie_id: str, new_chunks):
    """Merge new movie chunks into the global all_chunks.json."""
    import json
    from .config import PreprocessConfig as Cfg

    merged_path = Cfg.TEMPORAL_CHUNKS_DIR / "all_chunks.json"
    if merged_path.exists():
        data = json.loads(merged_path.read_text(encoding="utf-8"))
    else:
        data = {"metadata": {}, "chunks": []}

    # Remove old chunks for this movie
    data["chunks"] = [c for c in data["chunks"] if c["movie_id"] != movie_id]
    # Add new chunks
    data["chunks"].extend(new_chunks)
    data["metadata"]["total_chunks"] = len(data["chunks"])
    data["metadata"]["total_movies"] = len(set(c["movie_id"] for c in data["chunks"]))

    merged_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"  Updated all_chunks.json: {data['metadata']['total_chunks']} total chunks"
    )


def _fmt(seconds: float) -> str:
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        prog="preprocess_data",
        description="MovieRAG Data Preprocessing Pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # extract
    p_ext = sub.add_parser("extract", help="Extract keyframes from videos")
    p_ext.add_argument("--movie", type=str, help="Process single movie")
    p_ext.add_argument("--force", action="store_true", help="Overwrite existing")

    # build
    p_build = sub.add_parser("build", help="Build temporal chunks")
    p_build.add_argument("--movie", type=str, help="Process single movie")

    # index
    p_idx = sub.add_parser("index", help="Build FAISS + graph index")
    p_idx.add_argument("--movie", type=str, help="Process single movie")

    # all
    p_all = sub.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--movie", type=str, help="Process single movie")
    p_all.add_argument("--force", action="store_true", help="Overwrite existing")

    # ingest (NEW VIDEO)
    p_ingest = sub.add_parser("ingest", help="Ingest a new video into the system")
    p_ingest.add_argument("video", type=str, help="Path to video file")
    p_ingest.add_argument("--id", type=str, help="Movie ID (default: filename stem)")
    p_ingest.add_argument("--srt", type=str, help="Path to SRT subtitle file")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "all":
        cmd_all(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
