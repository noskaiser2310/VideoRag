import argparse
import json
import logging
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from groq import Groq
from tqdm import tqdm

logger = logging.getLogger("ingest_movie")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Determine project root and data paths dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_MOVIES_DIR = DATA_DIR / "Raw_Movies"
KEYFRAMES_DIR = DATA_DIR / "movienet_subset" / "shot_keyf"
TEMPORAL_CHUNKS_DIR = DATA_DIR / "temporal_chunks"
INDEX_DIR = DATA_DIR / "indexes"


class MovieIngester:
    def __init__(
        self, movie_filepath: str, imdb_id: str, srt_filepath: Optional[str] = None
    ):
        self.movie_filepath = Path(movie_filepath)
        self.imdb_id = imdb_id
        self.srt_filepath = Path(srt_filepath) if srt_filepath else None

        # Ensure directories exist
        self.movie_keyf_dir = KEYFRAMES_DIR / imdb_id
        self.movie_keyf_dir.mkdir(parents=True, exist_ok=True)
        TEMPORAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY chưa được thiết lập. Sẽ bỏ qua bước sinh Scene Metadata bằng LLM."
            )
            self.llm_client = None
        else:
            self.llm_client = Groq(api_key=api_key)

    def extract_frames(self, fps: str = "1/3") -> bool:
        """Sử dụng FFmpeg trích xuất hình ảnh (mặc định lấy 1 hình mỗi 3 giây)."""
        logger.info(f" [1/5] Bắt đầu trích xuất frames từ {self.movie_filepath}...")

        existing_frames = list(self.movie_keyf_dir.glob("*.jpg"))
        if len(existing_frames) > 50:
            logger.info(
                f" [1/5] Bỏ qua - Đã tìm thấy {len(existing_frames)} frames tại {self.movie_keyf_dir}."
            )
            return True

        if not self.movie_filepath.exists():
            logger.error(f" Không tìm thấy file video: {self.movie_filepath}")
            return False

        # Build FFmpeg command. Outputs shot_{index:04d}_img_0.jpg
        out_pattern = os.path.join(self.movie_keyf_dir, "shot_%04d_img_0.jpg")
        cmd = [
            "ffmpeg",
            "-i",
            str(self.movie_filepath),
            "-vf",
            f"fps={fps},scale=-1:360",
            "-qscale:v",
            "3",
            out_pattern,
            "-v",
            "warning",
            "-y",
        ]

        start_t = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_t
            frames_count = len(list(self.movie_keyf_dir.glob("*.jpg")))
            logger.info(
                f" [1/5] Hoàn tất trích xuất {frames_count} frames trong {elapsed:.1f}s."
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f" Lỗi FFmpeg: {e}")
            return False
        except FileNotFoundError:
            logger.error(" Không tìm thấy lệnh 'ffmpeg'. Vui lòng cài đặt FFmpeg.")
            return False

    def parse_srt(self) -> List[Dict]:
        """Đọc và parse file srt thành list các câu thoại kèm timestamp."""
        logger.info(f" [2/5] Bắt đầu xử lý subtitle (SRT)...")
        if not self.srt_filepath or not self.srt_filepath.exists():
            logger.warning(f"️ [2/5] Không có file SRT. Các đoạn sẽ bị trống text.")
            return []

        dialogues = []
        content = self.srt_filepath.read_text(encoding="utf-8", errors="replace")

        # Regex to parse standard SRT block:
        # 1\n00:00:01,000 --> 00:00:04,000\nHello World
        blocks = re.split(r"\n\s*\n", content)
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                time_line = lines[1]
                text = " ".join(lines[2:]).strip()
                # Clean html tags from text
                text = re.sub(r"<[^>]+>", "", text)

                match = re.search(
                    r"(\d{2}:\d{2}:\d{2}),\d{3} --> (\d{2}:\d{2}:\d{2}),\d{3}",
                    time_line,
                )
                if match:
                    start_t, end_t = match.groups()
                    dialogues.append({"start": start_t, "end": end_t, "text": text})
        logger.info(f" [2/5] Đã trích xuất {len(dialogues)} dòng thoại từ SRT.")
        return dialogues

    def chunk_dialogues(
        self, dialogues: List[Dict], chunk_length_secs: int = 120
    ) -> List[Dict]:
        """Gộp các câu thoại ngắn thành các Chunk dài vài phút để đưa vào RAG."""
        logger.info(" [3/5] Đóng gói các dòng thoại thành Temporal Chunks...")

        if not dialogues:
            # Fallback pseudo-chunks purely based on frames if no SRT
            return self._build_fake_chunks_from_frames(chunk_length_secs)

        def _time_to_secs(t_str: str) -> int:
            h, m, s = map(int, t_str.split(":"))
            return h * 3600 + m * 60 + s

        def _secs_to_time(s: int) -> str:
            return str(timedelta(seconds=s))

        chunks = []
        current_chunk = None
        chunk_idx = 1

        for d in dialogues:
            start_s = _time_to_secs(d["start"])

            if not current_chunk:
                current_chunk = {
                    "chunk": chunk_idx,
                    "start_time": d["start"],
                    "start_s": start_s,
                    "end_time": d["end"],
                    "dialogue": d["text"],
                }
            else:
                elapsed = start_s - current_chunk["start_s"]
                if elapsed < chunk_length_secs:
                    # Append strictly within the chunk length limit
                    current_chunk["end_time"] = d["end"]
                    current_chunk["dialogue"] += " " + d["text"]
                else:
                    # Finalize current chunk and start new one
                    chunks.append(
                        {
                            "id": f"{self.imdb_id}_chunk_{current_chunk['chunk']:04d}",
                            "movie_id": self.imdb_id,
                            "start_time": current_chunk["start_time"],
                            "end_time": current_chunk["end_time"],
                            "dialogue": current_chunk["dialogue"],
                        }
                    )
                    chunk_idx += 1
                    current_chunk = {
                        "chunk": chunk_idx,
                        "start_time": d["start"],
                        "start_s": start_s,
                        "end_time": d["end"],
                        "dialogue": d["text"],
                    }

        if current_chunk:
            chunks.append(
                {
                    "id": f"{self.imdb_id}_chunk_{current_chunk['chunk']:04d}",
                    "movie_id": self.imdb_id,
                    "start_time": current_chunk["start_time"],
                    "end_time": current_chunk["end_time"],
                    "dialogue": current_chunk["dialogue"],
                }
            )

        logger.info(
            f" [3/5] Gom nhóm thành công {len(chunks)} chunks (độ dài ~{chunk_length_secs}s mỗi chunk)."
        )
        return chunks

    def _build_fake_chunks_from_frames(self, chunk_length_secs: int) -> List[Dict]:
        """Nếu không có SRT, tự động phân rã phim thành các chunk dựa trên độ dài (giả định 120s/chunk)."""
        logger.warning(
            "Sử dụng fallback: Xây dựng Chunks mù (không có text) bằng thời gian cố định."
        )
        chunks = []
        # Count frames and assuming 1/3 fps, total duration = frames * 3
        frames = list(self.movie_keyf_dir.glob("*.jpg"))
        total_seconds = len(frames) * 3

        for i in range(0, total_seconds, chunk_length_secs):
            idx = (i // chunk_length_secs) + 1
            chunks.append(
                {
                    "id": f"{self.imdb_id}_chunk_{idx:04d}",
                    "movie_id": self.imdb_id,
                    "start_time": str(timedelta(seconds=i)),
                    "end_time": str(
                        timedelta(seconds=min(i + chunk_length_secs, total_seconds))
                    ),
                    "dialogue": "",
                }
            )
        return chunks

    def _llm_enrich_metadata(self, chunk_dialogue: str) -> Dict:
        """Sử dụng LLM trích xuất Description, Situation, Characters và Named Entities từ đoạn hội thoại."""
        if not self.llm_client or not chunk_dialogue.strip():
            return {
                "description": "",
                "situation": "",
                "characters": [],
                "entities": [],
            }

        prompt = (
            "Phân tích đoạn hội thoại phim dưới đây và trả về chuẩn xác theo định dạng JSON.\n"
            "Mục tiêu là trích xuất siêu dữ liệu (metadata) để làm Inverted Entity Index cho RAG.\n\n"
            "Các trường yêu cầu:\n"
            "- 'description': 1-2 câu tóm tắt nội dung đang nói.\n"
            "- 'situation': Chỉ 1-3 từ khóa ngắn gọn về hoàn cảnh (Vd: 'combat, dialogue, chase').\n"
            "- 'characters': Danh sách tên nhân vật được nhắc đến (phỏng đoán qua tên gọi/đại từ).\n"
            "- 'entities': Danh sách thực thể đặc trưng (Named Entities) như địa danh, vật phẩm quan trọng.\n\n"
            f'Đoạn thoại: "{chunk_dialogue[:1000]}"\n\n'
            "Trả về JSON trần (không markdown):"
        )
        try:
            res = self.llm_client.models.generate_content(
                model="moonshotai/kimi-k2-instruct",  # Cheap and fast
                contents=prompt,
            )
            text = res.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception:
            return {
                "description": "",
                "situation": "",
                "characters": [],
                "entities": [],
            }

    def enrich_and_save_chunks(self, chunks: List[Dict]):
        """Bổ sung metadata bằng LLM và lưu file JSON hoàn chỉnh."""
        logger.info(
            f" [4/5] Enriching Metadata using LLM (Extral Entity/Description) cho {len(chunks)} chunks..."
        )

        enriched_chunks = []
        # Progress bar
        for chunk in tqdm(chunks, desc="LLM Enriching"):
            meta = self._llm_enrich_metadata(chunk.get("dialogue", ""))

            chunk["description"] = meta.get("description", "")
            chunk["situation"] = meta.get("situation", "")
            chunk["characters"] = meta.get("characters", [])
            chunk["entities"] = meta.get("entities", [])

            # Map frames into the chunk (Time to Frame index heuristic based on fps=1/3)
            # Not fully perfect, but visually maps roughly.
            def t_to_s(t: str) -> int:
                h, m, s = map(int, t.split(":"))
                return h * 3600 + m * 60 + s

            start_s = t_to_s(chunk["start_time"])
            end_s = t_to_s(chunk["end_time"])

            # Since fps=1/3, frame index is approx seconds / 3
            first_frame_idx = start_s // 3
            last_frame_idx = end_s // 3

            chunk_frames = []
            for f_idx in range(first_frame_idx, last_frame_idx + 1):
                frame_name = (
                    f"shot_{f_idx + 1:04d}_img_0.jpg"  # +1 cuz indexing assumes 1
                )
                if (self.movie_keyf_dir / frame_name).exists():
                    chunk_frames.append(frame_name)

            chunk["keyframes"] = chunk_frames
            enriched_chunks.append(chunk)

        # Save to JSON
        output_json = TEMPORAL_CHUNKS_DIR / f"{self.imdb_id}_chunks.json"
        output_json.write_text(
            json.dumps(enriched_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f" [4/5] Đã lưu file Temporal Chunks ({output_json.name}).")
        return enriched_chunks

    def push_to_faiss(self, chunks: List[Dict]):
        """Nạp dữ liệu mới này thẳng vào bộ Indexer của Pipeline để tìm kiếm được ngay."""
        logger.info(
            f" [5/5] Re-indexing FAISS với {len(chunks)} chunks của {self.imdb_id}..."
        )

        # Load Existing Indexers
        from movierag.indexing.visual_indexer import VisualIndexer
        from movierag.indexing.knowledge_indexer import KnowledgeIndexer

        try:
            vi = VisualIndexer(str(INDEX_DIR))
            vi.load()
            ki = KnowledgeIndexer(str(INDEX_DIR))
            ki.load()

            # Prepare Knowledge Documents
            docs = []
            import copy
            from langchain_core.documents import Document

            for c in chunks:
                doc_text = f"Title: {self.imdb_id}\nDialogue: {c.get('dialogue', '')}\nDesc: {c.get('description', '')}"
                metadata = copy.deepcopy(c)
                # Keep metadata clean
                metadata.pop("keyframes", None)
                docs.append(Document(page_content=doc_text, metadata=metadata))

            # Add to text index
            if docs:
                ki.add_documents(docs)
                ki.save()

            # Add to visual index
            if vi.index_built:
                # Trích frame và index clip embeddings
                # We won't re-embed locally sequentially as it takes huge time. We just log instructions.
                logger.warning(
                    "Visual indexing cho thư mục mất nhiều thời gian do chạy CLIP locally."
                )
                logger.info(
                    f"Visual Index có thể được update bằng lệnh: VisualIndexer.build_index_from_dir({self.movie_keyf_dir})"
                )

            logger.info(" [5/5] Push FAISS Index thành công!")

        except Exception as e:
            logger.error(f" Indexing failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="End-To-End Ingestion Pipeline for a New Movie"
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Đường dẫn đến file video mp4/mkv",
    )
    parser.add_argument(
        "--id", "-i", type=str, required=True, help="Mã ID của phim (Vd: tt0120338)"
    )
    parser.add_argument(
        "--srt",
        "-s",
        type=str,
        required=False,
        help="Đường dẫn đến file phụ đề srt (Tuỳ chọn)",
    )

    args = parser.parse_args()

    ingester = MovieIngester(args.video, args.id, args.srt)

    # Run the Pipeline
    if ingester.extract_frames():
        raw_dialogues = ingester.parse_srt()
        raw_chunks = ingester.chunk_dialogues(raw_dialogues)
        enriched = ingester.enrich_and_save_chunks(raw_chunks)
        ingester.push_to_faiss(enriched)

        logger.info(
            f" TẤT CẢ HOÀN TẤT: Phim {args.id} đã được nạp độc lập vào hệ thống RAG."
        )


if __name__ == "__main__":
    main()
