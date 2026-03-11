import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

logger = logging.getLogger("eval_framework")

# 1. Định nghĩa cấu trúc Ground Truth Dataset
# Bố cục `eval_queries.json`:
# [
#     {
#         "query": "Rose và Jack gặp nhau lần đầu tiên gieo mình ở mũi tàu là đoạn nào?",
#         "expected_movie_id": "tt1285016",
#         "expected_shot_ids": ["shot_0143", "shot_0144"],
#         "expected_answer_keywords": ["Jack", "Rose", "mũi tàu", "cứu mạng"]
#     }
# ]


class MovieRAGEvaluator:
    def __init__(self, pipeline, llm_client, eval_file_path: str):
        self.pipeline = pipeline
        self.llm_client = llm_client
        self.eval_file_path = Path(eval_file_path)
        self.ground_truth = self._load_ground_truth()
        self.results = []

    def _load_ground_truth(self) -> List[Dict]:
        if not self.eval_file_path.exists():
            logger.warning(
                f"Không tìm thấy file eval dataset tại {self.eval_file_path}. Đang tạo file mẫu..."
            )
            sample_data = [
                {
                    "query": "Cảnh ông già lùn trần truồng đi trong tuyết là phim nào",
                    "expected_movie_id": "tt1127180",
                    "expected_shot_ids": ["shot_2667"],
                    "expected_answer_keywords": [
                        "Dr. Manhattan",
                        "Watchmen",
                        "sao Hỏa",
                        "cơ thể",
                    ],
                },
                {
                    "query": "Who is the mafia boss talking to the undertaker at the beginning of the movie?",
                    "expected_movie_id": "tt0068646",
                    "expected_shot_ids": ["shot_0003", "shot_0004", "shot_0005"],
                    "expected_answer_keywords": [
                        "Don Corleone",
                        "Bonasera",
                        "Godfather",
                        "wedding",
                    ],
                },
            ]
            self.eval_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_file_path.write_text(
                json.dumps(sample_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return sample_data

        try:
            return json.loads(self.eval_file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Lỗi đọc file eval: {e}")
            return []

    def _hit_k(
        self, top_k_results: List[Any], expected_movie: str, expected_shots: List[str]
    ) -> bool:
        """Đánh giá Hit@K cho Retrieval: Có trả về đúng phim và đúng cảnh không?"""
        for result in top_k_results:
            movie_id = (
                getattr(result, "movie_id", getattr(result.metadata, "movie_id", ""))
                if hasattr(result, "metadata")
                else getattr(result, "movie_id", "")
            )
            shot_id = (
                getattr(result, "metadata", {}).get("shot_id", "")
                if hasattr(result, "metadata")
                else ""
            )

            # Hit if movie matches AND (shot matches or expected_shots is empty)
            if movie_id == expected_movie:
                if not expected_shots:
                    return True

                # Check shot match
                for es in expected_shots:
                    if es in str(shot_id):
                        return True
        return False

    def _llm_as_judge(
        self, query: str, answer: str, expected_keywords: List[str]
    ) -> Dict[str, int]:
        """Dùng LLM (Kimi-K2) chấm điểm câu trả lời từ 1-5 theo 3 tiêu chí."""
        if not answer.strip():
            return {"accuracy": 1, "context": 1, "detail": 1}

        system_prompt = (
            "Bạn là một giám khảo AI (LLM-as-a-Judge) chuyên chấm điểm hệ thống RAG Điện Ảnh.\n"
            "Chấm điểm câu trả lời của hệ thống cho câu hỏi của User dựa trên thang điểm 1-5 (5 là tốt nhất).\n"
            "Các tiêu chí:\n"
            "1. Accuracy (Tính chính xác sự thật): Trả lời đúng trọng tâm câu hỏi, không viện dẫn sai phim/nhân vật.\n"
            "2. Context (Tính phù hợp bối cảnh): Có liên kết được hình ảnh/kịch bản vào câu trả lời để tạo sự tự nhiên không.\n"
            "3. Detail (Mức độ chi tiết): Cung cấp đủ thông tin sâu sắc hay trả lời cho có? Có đề cập các keywords bắt buộc không.\n\n"
            'CHỈ OUTPUT JSON THEO ĐỊNH DẠNG: {"accuracy": 5, "context": 4, "detail": 5}'
        )

        user_prompt = (
            f" Câu hỏi gốc: {query}\n"
            f" Keywords kỳ vọng (phải nhắc đến): {', '.join(expected_keywords)}\n\n"
            f" Câu trả lời của hệ thống RAG:\n{answer}\n\n"
            "Đánh giá và trả về JSON:"
        )

        try:
            response = self.llm_client.models.generate_content(
                model="moonshotai/kimi-k2-instruct",  # Fixed to primary cheap model
                contents=f"System: {system_prompt}\n\nUser: {user_prompt}",
            )
            # parse json from text
            text = response.text.replace("```json", "").replace("```", "").strip()
            score_dict = json.loads(text)
            return {
                "accuracy": score_dict.get("accuracy", 1),
                "context": score_dict.get("context", 1),
                "detail": score_dict.get("detail", 1),
            }
        except Exception as e:
            logger.error(f"LLM-as-judge failed: {e}")
            return {"accuracy": -1, "context": -1, "detail": -1}

    def run_eval(self):
        """Chạy toàn bộ quá trình tự động chấm điểm."""
        total_queries = len(self.ground_truth)
        hit_1_count = 0
        hit_5_count = 0

        logger.info(f"Bắt đầu chạy Evaluation Framework cho {total_queries} queries...")

        for i, gt in enumerate(self.ground_truth):
            query = gt["query"]
            expected_movie = gt.get("expected_movie_id", "")
            expected_shots = gt.get("expected_shot_ids", [])
            expected_kws = gt.get("expected_answer_keywords", [])

            logger.info(f"Đang đánh giá Query [{i + 1}/{total_queries}]: '{query}'")
            start_time = time.time()

            # --- 1. Gọi Pipeline System ---
            # Lưu ý: Pipeline của bạn hiện trả về chuỗi text và dùng generator, cần thu thập log
            response_chunks = []
            final_answer = ""
            visual_results_snapshot = []

            try:
                # Spy/Mock the visual search to capture results for Hit@K metric
                original_retrieve = self.pipeline.retrieve_visual

                def _spy_retrieve(q, k=6):
                    res = original_retrieve(q, k)
                    visual_results_snapshot.extend(res)
                    return res

                self.pipeline.retrieve_visual = _spy_retrieve

                # Run the actual pipeline (assuming Chat mode handles text queries)
                gen = self.pipeline.respond(
                    query, image_path=None, video_path=None, chat_history=[]
                )
                for chunk in gen:
                    response_chunks.append(chunk)

                # Cleanup spy
                self.pipeline.retrieve_visual = original_retrieve

                # Extract text (skipping <thought> blocks if formatted)
                final_answer = "".join(response_chunks)  # Simplified extraction

            except Exception as e:
                logger.error(f"Pipeline crashed on query '{query}': {e}")
                final_answer = f"ERROR: {e}"

            elapsed_time = time.time() - start_time

            # --- 2. Tính Point Metrics cho Retrieval ---
            hit_1 = self._hit_k(
                visual_results_snapshot[:1], expected_movie, expected_shots
            )
            hit_5 = self._hit_k(
                visual_results_snapshot[:5], expected_movie, expected_shots
            )

            if hit_1:
                hit_1_count += 1
            if hit_5:
                hit_5_count += 1

            # --- 3. Dùng LLM Judge chấm điểm đoạn Text ---
            llm_scores = self._llm_as_judge(query, final_answer, expected_kws)

            # Ghi nhận kết quả
            result_record = {
                "query": query,
                "latency_sec": round(elapsed_time, 2),
                "hit_1": hit_1,
                "hit_5": hit_5,
                "llm_scores": llm_scores,
                "answer_snippet": final_answer[-500:],  # Lưu lại 500 chữ cuối
            }
            self.results.append(result_record)

            logger.info(
                f"Kết quả Query {i + 1} | Hit@1: {hit_1} | Hit@5: {hit_5} | Scores: {llm_scores}"
            )

        # Compile Report
        report = {
            "total_queries": total_queries,
            "hit_1_accuracy": hit_1_count / total_queries if total_queries > 0 else 0,
            "hit_5_accuracy": hit_5_count / total_queries if total_queries > 0 else 0,
            "avg_accuracy_score": sum(
                r["llm_scores"].get("accuracy", 0) for r in self.results
            )
            / total_queries
            if total_queries > 0
            else 0,
            "avg_latency": sum(r["latency_sec"] for r in self.results) / total_queries
            if total_queries > 0
            else 0,
            "details": self.results,
        }

        report_path = self.eval_file_path.parent / "eval_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"\n Evaluation hoàn tất! Báo cáo lưu tại: {report_path}")
        logger.info(
            f" Tổng quan: Hit@1 = {report['hit_1_accuracy'] * 100:.1f}%, LLM Accuracy = {report['avg_accuracy_score']:.1f}/5.0"
        )

        return report


# How to use:
# from movierag.evaluation.eval_framework import MovieRAGEvaluator
# evaluator = MovieRAGEvaluator(pipeline, llm_client, "D:/Study/School/project_ky4/src/movierag/evaluation/eval_queries.json")
# evaluator.run_eval()
