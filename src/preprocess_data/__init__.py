"""
preprocess_data — Unified Data Processing Pipeline for MovieRAG

Sub-modules:
    video/      Download videos, extract keyframes, detect shots
    dataset/    Load MovieNet/MovieGraphs, build unified dataset
    temporal/   Build 5-layer temporal chunks, parse subtitles
    indexing/    Build FAISS index, enrich knowledge graph

CLI:
    python -m preprocess_data download  --movie tt0120338
    python -m preprocess_data extract   --movie tt0120338
    python -m preprocess_data build     --movie tt0120338
    python -m preprocess_data index     --movie tt0120338
    python -m preprocess_data all       --movie tt0120338
"""

from .config import PreprocessConfig
