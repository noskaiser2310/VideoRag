"""
Agentic VideoRAG Pipeline
==========================
Multi-agent pipeline: Contextualize → Route Intent → Retrieve → Grade → Generate.
Inspired by RagLaw LangGraph + SceneRAG + DrVideo patterns.
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

logger = logging.getLogger(__name__)

MAX_REWRITE = 3


class QueryIntent(str, Enum):
    """4-way intent classification for MovieRAG queries."""

    VISUAL = "VISUAL"  # Needs frame/image retrieval
    KNOWLEDGE = "KNOWLEDGE"  # Text-only (cast, plot, metadata)
    MULTIMODAL = "MULTIMODAL"  # Both visual + knowledge
    CHAT = "CHAT"  # General conversation, no retrieval
    MACRO_KNOWLEDGE = "MACRO_KNOWLEDGE"  # Plot summary, big picture synopsis
    DIALOG = "DIALOG"  # Dialog path: Subtitle/quote search


class AgenticVideoRAGPipeline:
    """
    Multi-agent pipeline for MovieRAG.

    Flow:
        contextualize → route_intent → [retrieve_visual | retrieve_knowledge | both]
        → grade → [rewrite × 3] → generate → format_response
    """

    def __init__(
        self,
        visual_indexer=None,
        knowledge_indexer=None,
        dialogue_indexer=None,
        llm_generator=None,
        model_id: str = "gemma-3-27b-it",
        api_key: Optional[str] = None,
    ):
        self.visual_indexer = visual_indexer
        self.knowledge_indexer = knowledge_indexer
        self.dialogue_indexer = dialogue_indexer
        self.graph_indexer = None
        self.llm_generator = llm_generator
        self.model_id = model_id

        # Initialize LLM client for routing/grading
        self._llm_client = None
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._init_llm_client()

        # Grader
        from movierag.pipeline.grader import DocumentGrader

        self.grader = DocumentGrader(llm_client=self._llm_client, model_id=model_id)

        # GraphRAG Neo4j Driver Connection
        self._neo4j_driver = None
        if GraphDatabase:
            try:
                self._neo4j_driver = GraphDatabase.driver(
                    "bolt://localhost:7688", auth=("neo4j", "movierag123")
                )
                logger.info(" Connected to Neo4j Graph Database for True GraphRAG.")
            except Exception as e:
                logger.warning(
                    f"Neo4j connection failed. Is docker-compose running? {e}"
                )

        # Fallback Local Graph Indexer
        try:
            from movierag.indexing.graph_indexer import GraphIndexer
            self.graph_indexer = GraphIndexer(index_dir="data/indexes")
            self.graph_indexer.load()
        except Exception as e:
            logger.warning(f"Could not load local GraphIndexer fallback: {e}")

    def _init_llm_client(self):
        try:
            from movierag.generation.universal_client import UniversalLLMClient

            self._llm_client = UniversalLLMClient(model_id=self.model_id)
        except Exception as e:
            logger.warning(f"Could not initialize pipeline UniversalLLMClient: {e}")

    #  Node 1: Contextualize 

    def contextualize(
        self, query: str, history: List[Dict], has_media: bool = False
    ) -> str:
        """Rewrite query using chat history for standalone context.

        Only rewrites for pure text queries. When image/video is present,
        the original query is preserved to avoid LLM misinterpreting
        the user's intent without seeing the media.
        """
        # Skip rewrite for multimodal queries — media provides its own context
        if has_media:
            return query

        if not history or not self._llm_client:
            return query

        # Only use last 3 turns for context
        recent = history[-6:]  # 3 user + 3 assistant messages
        history_text = ""
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                history_text += f"{role}: {content}\n"

        if not history_text.strip():
            return query

        try:
            prompt = (
                f"Dựa trên lịch sử chat:\n{history_text}\n\n"
                f"Câu hỏi mới: {query}\n\n"
                f"Viết lại câu hỏi thành câu độc lập, đầy đủ ngữ cảnh. "
                f"Nếu đã đầy đủ, giữ nguyên. Chỉ trả về câu hỏi mới."
            )
            response = self._llm_client.models.generate_content(
                model=self.model_id, contents=prompt
            )
            rewritten = response.text.strip()
            if rewritten and len(rewritten) < 500:
                logger.info(f"[Contextualize] '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Contextualize failed: {e}")

        return query

    #  Node 2: Route Intent & Extract Explicit Movie 

    def route_intent_and_movie(
        self, query: str, has_image: bool = False
    ) -> tuple[QueryIntent, str]:
        """Classify user intent AND extract explicit movie name using language model."""
        if has_image:
            logger.info("[Route] Image detected -> MULTIMODAL")
            # We still might want to extract movie name if they gave one with the image!
            # Let's run the LLM even if has_image is true, just force the intent to MULTIMODAL.

        if not self._llm_client:
            return (QueryIntent.MULTIMODAL if has_image else QueryIntent.KNOWLEDGE), ""

        prompt = (
            "Bạn là một hệ thống phân tích câu hỏi của người dùng về ĐIỆN ẢNH.\n"
            "Nhiệm vụ 1: Phân loại ý định người dùng thành 1 trong 4 loại:\n"
            "- VISUAL: Yêu cầu xem hình ảnh, xem mặt mũi, xem cách bố trí, hình dáng, màu sắc.\n"
            "- KNOWLEDGE: Tìm thông tin chữ (Ai đóng vai?, Tóm tắt phim, Diễn viên là ai?, Quote này của ai?).\n"
            "- MULTIMODAL: Yêu cầu phức tạp cần đọc cả nội dung LẪN xem hình ảnh để kết luận.\n"
            "- CHAT: Trò chuyện bình thường, chào hỏi và trả lời thân thiện.\n\n"
            "Nhiệm vụ 2: RÚT TRÍCH TÊN PHIM (EXPLICIT MOVIE NAME)\n"
            "- Nếu người dùng CỐ TÌNH NHẮC ĐẾN tên một bộ phim cụ thể (Ví dụ: 'Trong phim Titanic...', 'Watchmen có cảnh nào...'), hãy rút trích tên phim đó ra.\n"
            "- Nếu câu hỏi chung chung không nhắc rõ tên phim (VD: 'Cảnh ông già lùn trần truồng là phim nào?'), để trống.\n\n"
            "CHỈ OUTPUT JSON THEO ĐỊNH DẠNG SAU, KHÔNG GIẢI THÍCH:\n"
            '{"intent": "KNOWLEDGE", "explicit_movie": "Titanic"}\n\n'
            f"Câu hỏi: {query}\n"
            "JSON:"
        )
        try:
            res = self._llm_client.models.generate_content(
                model="moonshotai/kimi-k2-instruct", contents=prompt
            )
            # parse json
            import json as _json
            import re as _re

            text = res.text.strip()
            # Clean markdown code blocks
            text = _re.sub(r"```json\n|\n```|```", "", text).strip()
            data = _json.loads(text)

            intent_str = data.get("intent", "").upper()
            explicit_movie = data.get("explicit_movie", "").strip()

            # If user uploaded an image, override intent to MULTIMODAL regardless of text
            final_intent = (
                QueryIntent.MULTIMODAL if has_image else QueryIntent.KNOWLEDGE
            )

            if not has_image:
                if "VISUAL" in intent_str:
                    final_intent = QueryIntent.VISUAL
                elif "CHAT" in intent_str:
                    final_intent = QueryIntent.CHAT
                elif "MULTIMODAL" in intent_str:
                    final_intent = QueryIntent.MULTIMODAL

            logger.info(
                f"[Route] '{query}' -> Intent: {final_intent.name}, Movie: '{explicit_movie}'"
            )
            return final_intent, explicit_movie

        except Exception as e:
            logger.warning(f"Intent & Movie routing LLM failed: {e}")
            return (QueryIntent.MULTIMODAL if has_image else QueryIntent.KNOWLEDGE), ""

    #  Node 3: Retrieve 

    def retrieve_visual(self, query: str, k: int = 6) -> list:
        """Retrieve visual frames using CLIP FAISS."""
        if not self.visual_indexer:
            return []
        if not (
            hasattr(self.visual_indexer, "_is_loaded")
            and self.visual_indexer._is_loaded
        ):
            return []

        try:
            if hasattr(self.visual_indexer, "hierarchical_search"):
                res = self.visual_indexer.hierarchical_search(query, k=k, scene_k=5)
            else:
                res = self.visual_indexer.search_by_text(query, k=k)

            # Filter low confidence
            filtered_res = [r for r in res if getattr(r, "score", 0.0) >= 0.36]
            return filtered_res[:k]
        except Exception as e:
            logger.error(f"Visual retrieval failed: {e}")
            return []

    def retrieve_visual_by_image(self, image_path: str, k: int = 6) -> list:
        """Retrieve visual frames by image similarity (FAISS direct)."""
        if not self.visual_indexer:
            return []
        try:
            results = self.visual_indexer.search_by_image(
                image_path, k=k, exclude_same=False
            )
            return results[:k]
        except Exception as e:
            logger.error(f"Visual image retrieval failed: {e}")
            return []

    def retrieve_knowledge(
        self, query: str, k: int = 5, movie_id: Optional[str] = None
    ) -> list:
        """Retrieve text documents from knowledge FAISS index."""
        if not self.knowledge_indexer:
            return []
        # Let exceptions bubble up to the grading loop so they can be captured in thoughts
        return self.knowledge_indexer.search(query, k=k, movie_id=movie_id)

    def query_graph(self, query: str) -> str:
        """Run Graph search: Neo4j if available, otherwise local NetworkX."""
        result_text = ""
        # 1. Try Neo4j first
        if self._neo4j_driver:
            try:
                # If the query is an actual Cypher query
                if "MATCH" in query.upper() or "RETURN" in query.upper():
                    # Neo4j 5.x compatible execute_query
                    from neo4j import RoutingControl
                    records, summary, keys = self._neo4j_driver.execute_query(
                        query, routing_=RoutingControl.READ
                    )
                    if records:
                        result_text = "\n".join([str(r.data()) for r in records[:5]])
                        return f"Neo4j Results:\n{result_text}"
                    return "Neo4j query returned no results."
            except Exception as e:
                logger.warning(f"Neo4j cypher execution failed: {e}. Falling back to local.")
        
        # 2. Fallback to Local NetworkX Graph Indexer
        if getattr(self, "graph_indexer", None):
            try:
                res = self.graph_indexer.search(query, k=5)
                if res:
                    lines = [f"Clip: {r.get('clip_id')} | {r.get('text', '')[:100]}" for r in res]
                    return "Local Graph Results:\n" + "\n".join(lines)
                return "Local Graph query returned no results."
            except Exception as e:
                logger.warning(f"Local graph search failed: {e}")
                
        return "[Graph Database not connected or failed]"

    #  Node 5: Generate 

    def generate_answer(
        self,
        query: str,
        intent: QueryIntent,
        visual_results: list,
        knowledge_results: list,
        history: List[Dict],
    ) -> str:
        """Generate final answer using LLM."""
        if intent == QueryIntent.CHAT:
            return self._generate_chat(query, history)

        if not self.llm_generator:
            return "️ LLM chưa được khởi tạo."

        try:
            # Map intent to route for LLM prompt
            route = self._intent_to_route(intent)
            return self.llm_generator.generate_answer(
                query=query,
                context_results=knowledge_results,
                visual_results=visual_results,
                history=history,
                route=route,
            )
        except Exception as e:
            return f"️ LLM Error: {e}"

    def _generate_chat(self, query: str, history: List[Dict]) -> str:
        """Direct chat response without retrieval."""
        if not self._llm_client:
            return "Xin chào! Tôi là MovieRAG, hệ thống tìm kiếm phim thông minh. Bạn có thể hỏi về phim, tìm cảnh, hoặc tra cứu thông tin diễn viên."
        try:
            prompt = (
                "Bạn là MovieRAG, trợ lý tìm kiếm phim thông minh. "
                "Trả lời ngắn gọn, thân thiện.\n\n"
                f"Câu hỏi: {query}"
            )
            response = self._llm_client.models.generate_content(
                model=self.model_id, contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Xin chào! Tôi là MovieRAG. Hãy hỏi tôi về phim! (Error: {e})"

    def _intent_to_route(self, intent: QueryIntent):
        """Map QueryIntent to old QueryRoute for backward compatibility."""
        try:
            from movierag.routing.query_router import QueryRoute

            mapping = {
                QueryIntent.VISUAL: QueryRoute.FACTUAL,
                QueryIntent.KNOWLEDGE: QueryRoute.FACTUAL,
                QueryIntent.MULTIMODAL: QueryRoute.REASONING,
                QueryIntent.CHAT: QueryRoute.FACTUAL,
            }
            return mapping.get(intent, QueryRoute.FACTUAL)
        except ImportError:
            return None

    #  Main Entry Point 

    def respond(
        self,
        query: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Full agentic pipeline: Contextualize → Route → Retrieve → Grade → Generate.

        Returns:
            {
                "answer": str,
                "intent": str,
                "visual_results": list,
                "knowledge_results": list,
                "keyframe_paths": list,
                "thoughts": list,  # Agent reasoning trace
            }
        """
        history = history or []
        thoughts = []

        #  Step 1: Contextualize (text-only, skip for multimodal) 
        has_media = image_path is not None or video_path is not None
        contextualized = self.contextualize(query, history, has_media=has_media)
        if contextualized != query:
            thoughts.append(f" Query rewritten: '{contextualized}'")

        #  Step 2: Route Intent & Extract Explicit Movie 
        intent, explicit_movie_name = self.route_intent_and_movie(
            contextualized, has_image=image_path is not None or video_path is not None
        )
        thoughts.append(f" Intent: **{intent.value}**")

        import tools.movie_resolver as resolver
        
        identified_movie = None
        if explicit_movie_name:
            identified_movie = resolver.resolve_movie_name(explicit_movie_name)
            if identified_movie:
                thoughts.append(
                    f" Explicit Movie Identified: '{explicit_movie_name}' -> ID: {identified_movie}"
                )

        #  Step 3: Swarm Verification Loop (No Tools for Gemma) 
        max_iterations = 3
        current_queries = [contextualized]
        rewrite_count = 0

        visual_results = []
        knowledge_results = []

        #  Step 2.5: Pre-Loop Visual Processing & VLM Analysis 
        vlm_analysis_text = ""
        movie_meta_cache = {}
        identified_movie = None
        target_media_for_vlm = image_path

        if video_path:
            thoughts.append(f" Đang trích xuất khung hình từ video: {video_path}")
            try:
                import cv2
                import tempfile
                import os as _os

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                n_samples = min(8, max(1, total_frames))
                sample_positions = [
                    int(i * total_frames / n_samples) for i in range(n_samples)
                ]

                frame_results_all = []
                extracted_frame_paths = []  # Keep actual video frames for VLM
                movie_vote = {}  # majority voting: movie_id -> count
                exact_match_movie = None
                exact_match_score = 0.0

                for pos in sample_positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    tmp_f = tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False, dir=_os.environ.get("TEMP", "/tmp")
                    )
                    cv2.imwrite(tmp_f.name, frame)
                    tmp_f.close()
                    frame_matches = self.retrieve_visual_by_image(tmp_f.name, k=3)
                    for m in frame_matches:
                        mid = getattr(m, "movie_id", "")
                        score = getattr(m, "score", 0.0)
                        if score >= 0.90 and not exact_match_movie:
                            exact_match_movie = mid
                            exact_match_score = score
                            
                        movie_vote[mid] = movie_vote.get(mid, 0) + 1
                        frame_results_all.append(m)
                    extracted_frame_paths.append(tmp_f.name)  # Keep for VLM
                cap.release()

                # Keep the middle frame for VLM, clean up the rest
                vlm_frame_path = None
                for idx, fp in enumerate(extracted_frame_paths):
                    if idx == len(extracted_frame_paths) // 2:
                        vlm_frame_path = fp  # Keep middle frame
                    else:
                        try:
                            _os.unlink(fp)
                        except Exception:
                            pass

                if movie_vote:
                    if exact_match_movie:
                        identified_movie = exact_match_movie
                        movie_title = identified_movie
                        for r in frame_results_all:
                            if getattr(r, "movie_id", "") == identified_movie:
                                meta = getattr(r, "metadata", {})
                                if isinstance(meta, dict) and "title" in meta:
                                    movie_title = meta["title"]
                                    break
                        thoughts.append(
                            f"⚡ Tín hiệu Video cực mạnh ({exact_match_score:.2f} > 0.90). Không xét majority vote: phim được nhận diện là **{movie_title}** ({identified_movie})"
                        )
                    else:
                        identified_movie = max(movie_vote, key=movie_vote.get)
                        movie_title = identified_movie
                        
                        for r in frame_results_all:
                            if getattr(r, "movie_id", "") == identified_movie:
                                meta = getattr(r, "metadata", {})
                                if isinstance(meta, dict) and "title" in meta:
                                    movie_title = meta["title"]
                                    break
                                    
                        thoughts.append(
                            f"🎩 Video Majority Voting: phim được nhận diện là **{movie_title}** ({identified_movie})"
                        )

                    # Deduplicate visual results
                    seen_ids = set([getattr(x, "id", str(x)) for x in visual_results])
                    for r in frame_results_all:
                        if getattr(r, "movie_id", "") == identified_movie:
                            _key = getattr(r, "id", str(r))
                            if _key not in seen_ids:
                                visual_results.append(r)
                                seen_ids.add(_key)

                    current_queries = [f"{contextualized} {movie_title}"]
                    # Use ACTUAL extracted video frame for VLM (not DB keyframe!)
                    if vlm_frame_path:
                        target_media_for_vlm = vlm_frame_path
                    elif frame_results_all:
                        # Fallback to DB keyframe only if no extracted frame
                        target_media_for_vlm = getattr(
                            frame_results_all[0],
                            "path",
                            getattr(frame_results_all[0].metadata, "path", None),
                        )
                else:
                    thoughts.append("️ Không thể nhận diện phim từ video.")
            except Exception as e:
                thoughts.append(f" Lỗi xử lý video: {e}")

        elif image_path:
            thoughts.append(f"️ Tìm kiếm Visual gốc cho ảnh tải lên...")
            vr = self.retrieve_visual_by_image(image_path, k=5)
            # Find exact matches (>0.90 score) to skip majority vote
            exact_match_movie = None
            exact_match_score = 0.0
            
            for r in vr:
                score = getattr(r, "score", 0.0)
                if score >= 0.90:
                    exact_match_movie = getattr(r, "movie_id", "")
                    exact_match_score = score
                    break
                    
            if exact_match_movie:
                identified_movie = exact_match_movie
                movie_title = identified_movie
                # Get proper title
                for r in vr:
                    if getattr(r, "movie_id", "") == identified_movie:
                        meta = getattr(r, "metadata", {})
                        if isinstance(meta, dict) and "title" in meta:
                            movie_title = meta["title"]
                            break
                            
                thoughts.append(
                    f"️ Tín hiệu Visual cực mạnh ({exact_match_score:.2f} > 0.90). Không xét majority vote: phim được nhận diện là **{movie_title}** ({identified_movie})"
                )
            else:
                movie_vote = {}
                for r in vr:
                    mid = getattr(r, "movie_id", "")
                    if mid:
                        movie_vote[mid] = movie_vote.get(mid, 0) + 1
                if movie_vote:
                    identified_movie = max(movie_vote, key=movie_vote.get)
                    movie_title = identified_movie
                    for r in vr:
                        if getattr(r, "movie_id", "") == identified_movie:
                            meta = getattr(r, "metadata", {})
                            if isinstance(meta, dict) and "title" in meta:
                                movie_title = meta["title"]
                                break
                                
                    thoughts.append(
                        f" Image Majority Voting: phim được nhận diện là **{movie_title}** ({identified_movie})"
                    )

            # Seed visual results (deduplicate by id)
            seen_ids = set([getattr(x, "id", str(x)) for x in visual_results])
            for r in vr:
                _key = getattr(r, "id", str(r))
                if _key not in seen_ids:
                    visual_results.append(r)
                    seen_ids.add(_key)

        #  Step 2.6: Extract enriched metadata from FAISS matched shots 
        matched_shots_context = ""
        if visual_results:
            lines = []
            for i, v in enumerate(visual_results[:8]):  # Top 8 matches
                meta = getattr(v, "metadata", {})
                mid = getattr(v, "movie_id", meta.get("movie_id", ""))
                shot_id = getattr(v, "id", meta.get("id", ""))
                score = getattr(v, "score", 0.0)
                start_t = meta.get("start_time", "")
                end_t = meta.get("end_time", "")
                desc = meta.get("description", "")
                dial = meta.get("dialogue_text", "")
                chars = meta.get("characters", [])
                situation = meta.get("situation", "")
                title = meta.get("title", mid)

                line = f"[{i + 1}] {mid}›{shot_id} · {score:.2f}"
                if title and title != mid:
                    line += f" | Title: {title}"
                if start_t:
                    line += f" | Time: {start_t}→{end_t}"
                if situation:
                    line += f" | Situation: {situation}"
                if desc:
                    line += f" | Desc: {desc[:120]}"
                if chars:
                    line += f" | Characters: {', '.join(chars[:5])}"
                if dial:
                    line += f" | Dialogue: {dial[:80]}"
                lines.append(line)

            matched_shots_context = "\n".join(lines)
            thoughts.append(
                f" Trích xuất metadata trực tiếp từ {len(lines)} shots khớp nhất"
            )

        #  Step 2.7: VLM Analysis — ONLY for image queries 
        # Video: VLM only sees 1 frame → incomplete analysis. FAISS majority voting + metadata is enough.
        # Image: VLM sees the full uploaded image → useful for scene context + verification.
        if target_media_for_vlm and image_path and not video_path:
            try:
                import json as _json
                from pathlib import Path as _Path

                def _load_movie_meta_vlm(movie_id: str) -> dict:
                    _META_DIRS = [
                        _Path("data/movienet_subset/meta"),
                        _Path("data/unified_dataset/meta"),
                        _Path("../data/movienet_subset/meta"),
                        _Path("../data/unified_dataset/meta"),
                    ]
                    for d in _META_DIRS:
                        p = d / f"{movie_id}.json"
                        if p.exists():
                            try:
                                return _json.loads(p.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                    return {}

                # Load meta for identified movie
                meta = {}
                title = identified_movie or ""
                if identified_movie:
                    meta = _load_movie_meta_vlm(identified_movie)
                    if meta:
                        movie_meta_cache[identified_movie] = meta
                        title = meta.get("title", identified_movie)

                thoughts.append(
                    "️ Gọi VLM (Vision Model) phân tích bối cảnh và xác minh shots..."
                )

                # VLM prompt: describe scene + verify against FAISS matches
                vlm_prompt = (
                    "Hãy quan sát thật kỹ bức ảnh/frame này và thực hiện 2 nhiệm vụ:\n\n"
                    "1. MÔ TẢ CHI TIẾT: Nhân vật (ngoại hình, trang phục, biểu cảm), "
                    "bối cảnh (nội/ngoại, ánh sáng, vật thể), hành động đang diễn ra.\n\n"
                )
                if matched_shots_context:
                    vlm_prompt += (
                        "2. XÁC MINH: Hệ thống FAISS đã tìm được các shots sau. "
                        "Dựa vào nội dung ảnh, hãy cho biết shot nào khớp nhất (hoặc KHÔNG khớp):\n"
                        f"{matched_shots_context}\n\n"
                        "Trả lời: [MÔ TẢ chi tiết], sau đó [SHOT PHÙ HỢP NHẤT: số thứ tự]"
                    )
                else:
                    vlm_prompt += (
                        "Mô tả cực kì chi tiết cấu trúc hạt nhân của bức ảnh "
                        "để làm Query tìm kiếm Visual Search."
                    )

                vlm_res = self._llm_client.generate_vision_content(
                    prompt=vlm_prompt, image_path=target_media_for_vlm
                )
                vlm_analysis_text = (
                    f"️ KẾT QUẢ VLM (Scene Context + Verification):\n{vlm_res}\n\n"
                )
                thoughts.append(f" VLM Response: {vlm_res[:200]}...")

                #  Check VLM-FAISS conflict: does VLM description match FAISS movie? 
                vlm_conflict_warning = ""
                if vlm_res and identified_movie and matched_shots_context:
                    try:
                        conflict_prompt = (
                            f"VLM mô tả ảnh: {vlm_res[:300]}\n"
                            f"FAISS cho rằng đây là phim: {title} (ID: {identified_movie})\n\n"
                            f"Hỏi: Nội dung VLM mô tả có KHỚP với phim {title} không?\n"
                            f"Trả lời CHỈ MỘT từ: MATCH hoặc MISMATCH"
                        )
                        conflict_res = self._llm_client.models.generate_content(
                            model=self.model_id, contents=conflict_prompt
                        )
                        conflict_text = conflict_res.text.strip().upper()
                        if "MISMATCH" in conflict_text:
                            vlm_conflict_warning = (
                                f"️ CẢNH BÁO: VLM phát hiện nội dung ảnh KHÔNG KHỚP với phim {title}. "
                                f"Ảnh có thể không nằm trong cơ sở dữ liệu. "
                                f"Hãy trả lời dựa trên mô tả VLM thay vì FAISS results.\n"
                            )
                            thoughts.append(
                                f"️ VLM-FAISS CONFLICT: VLM mô tả không khớp với {title}! Ảnh có thể không trong DB."
                            )
                    except Exception:
                        pass  # Conflict check failed, proceed normally

                #  Distill VLM output into clean search keywords 
                if vlm_res and len(vlm_res.split()) > 1:
                    try:
                        distill_prompt = (
                            f"Từ mô tả VLM sau, trích xuất 5-10 từ khóa tìm kiếm hình ảnh (tiếng Anh, ngắn gọn, phân cách bằng dấu cách).\n"
                            f"KHÔNG giải thích, chỉ trả về từ khóa.\n\n"
                            f"VLM: {vlm_res[:500]}\n\nKeywords:"
                        )
                        distill_res = self._llm_client.models.generate_content(
                            model=self.model_id, contents=distill_prompt
                        )
                        keywords = distill_res.text.strip()[:150]
                    except Exception:
                        keywords = " ".join(vlm_res.split()[:10])

                    injected_query = f"{title} {keywords}"
                    thoughts.append(f" Bơm VLM keywords: '{injected_query}'")
                    current_queries.append(injected_query)
                    if len(query.split()) <= 4 and intent == QueryIntent.MULTIMODAL:
                        query = injected_query

            except Exception as e:
                thoughts.append(f"️ VLM Analysis failed: {e}")
                vlm_res = ""
        else:
            vlm_conflict_warning = ""
            vlm_res = ""

        #  Step 2.8: LLM Context Booster (for Video / non-VLM cases) 
        # Focus on generating keywords based on the video context or explicitly mentioned movie
        vlm_res = locals().get("vlm_res", "")
        if not vlm_res and (matched_shots_context or explicit_movie_name):
            try:
                thoughts.append(
                    " Gọi LLM Context Booster: Tổng hợp metadata và ngữ cảnh thành truy vấn nâng cao..."
                )
                boost_prompt = (
                    "Bạn là Query Booster. Dựa trên thông tin đầu vào (Tên phim hoặc Dữ liệu FAISS Video), "
                    "hãy trích xuất 2-4 TỪ KHÓA TÌM KIẾM (tiếng Anh) ngắn gọn, sắc bén để tìm kiếm trong CSDL RAG.\n"
                    "Trả về định dạng các cụm từ cách nhau bởi dấu `|`. KHÔNG giải thích thêm.\n"
                    "Ví dụ: rose titanic | jack drawing | door sinking\n\n"
                    f"Câu hỏi gốc: {query}\n"
                )
                if explicit_movie_name:
                    boost_prompt += (
                        f"Bối cảnh (Tên phim trực tiếp): {explicit_movie_name}\n"
                    )
                if matched_shots_context:
                    boost_prompt += f"Dữ liệu FAISS Video:\n{matched_shots_context}\n\n"

                boost_prompt += "Output:"
                boost_res = self._llm_client.models.generate_content(
                    model=self.model_id, contents=boost_prompt
                )
                boost_text = boost_res.text.strip()

                # Strip <think>...</think> reasoning tokens (Qwen3, etc.)
                import re as _re

                boost_text = _re.sub(r"<think>[\s\S]*?</think>", "", boost_text).strip()

                new_queries = [
                    q.strip() for q in boost_text.split("|") if len(q.strip()) > 5
                ]
                if new_queries:
                    current_queries.extend(new_queries)
                    thoughts.append(f" Bơm Metadata Keywords: {new_queries}")
            except Exception as e:
                thoughts.append(f"️ LLM Context Booster failed: {e}")

        if not vlm_res:
            vlm_conflict_warning = ""

        # For MULTIMODAL with image/video: FAISS visual = primary evidence, limit retries
        effective_max_iter = max_iterations
        if intent == QueryIntent.MULTIMODAL and (image_path or video_path):
            effective_max_iter = (
                1  # Visual matching IS the answer, no need to retry text
            )
            thoughts.append(
                " Multimodal: FAISS visual là evidence chính, giới hạn 1 vòng Verifier."
            )

        if intent == QueryIntent.CHAT:
            thoughts.append(" Chat mode — no retrieval needed")
        else:
            for iteration in range(effective_max_iter):
                lore_report = ""
                visual_report = ""

                # Deduplication dictionaries for this iteration
                k_results_dict = {
                    getattr(k, "id", str(k)): k for k in knowledge_results
                }
                v_results_dict = {getattr(v, "id", str(v)): v for v in visual_results}

                thoughts.append(
                    f" Vòng {iteration + 1}: Chạy tìm kiếm cho {len(current_queries)} truy vấn đồng thời..."
                )

                for q in current_queries:
                    #  Lore Execution (Static Python Calls) 
                    if intent in (
                        QueryIntent.KNOWLEDGE,
                        QueryIntent.MULTIMODAL,
                        QueryIntent.MACRO_KNOWLEDGE,
                        QueryIntent.DIALOG,
                    ):
                        # Bỏ qua tìm text FAISS nếu có ảnh VÀ câu hỏi quá ngắn/chung chung (VD: "Ai đây?", "Đây là phim gì?")
                        skip_text_search = False
                        if image_path and len(q.split()) <= 4:
                            skip_text_search = True
                            thoughts.append(
                                f" Bỏ qua tìm text cho '{q}' vì có ảnh và câu hỏi chung chung."
                            )

                        if not skip_text_search:
                            thoughts.append(
                                f" Gọi LoreAgent truy xuất Đồ thị và Kịch bản cho: '{q}'"
                            )
                            # If VLM injected a new query (i.e. not the main query), drop the strict movie filter so it can find correct movies
                            movie_filter = (
                                identified_movie if q == current_queries[0] else None
                            )
                            try:
                                kr = self.retrieve_knowledge(
                                    q, k=4, movie_id=movie_filter
                                )
                                for k in kr:
                                    k_results_dict[getattr(k, "id", str(k))] = k
                                lore_report += (
                                    f"Knowledge Docs cho '{q}': {len(kr)} found.\n"
                                )
                            except Exception as e:
                                thoughts.append(f"️ Lỗi Search Text FAISS: {e}")
                                logger.error(
                                    f"Knowledge search error: {e}", exc_info=True
                                )

                        dia_idx = getattr(self, "dialogue_indexer", None)
                        if dia_idx:
                            movie_filter = (
                                identified_movie if q == current_queries[0] else None
                            )
                            try:
                                dr = dia_idx.search(q, k=2, movie_id=movie_filter)
                            except TypeError:
                                # Fallback if dialogue_indexer hasn't fully reloaded the new signature in worker thread
                                dr = dia_idx.search(q, k=2)
                            except Exception as e:
                                logger.error(f"Dialogue search error: {e}")
                                dr = []

                            if dr:
                                lore_report += f"Dialogue Results cho '{q}':\n"
                                for i, r in enumerate(dr):
                                    lore_report += f"[{i + 1}] {r.get('movie_id', '')} - {r.get('start_time', 0)}: '{r.get('text', '')}'\n"

                    #  Visual Execution (Static Python Calls) 
                    if intent in (QueryIntent.VISUAL, QueryIntent.MULTIMODAL):
                        thoughts.append(
                            f"️ Gọi VisualAgent tìm kiếm Database Hình ảnh cho: '{q}'"
                        )
                        # If this is the original query and we have an image, search by image.
                        # If the VLM expanded/corrected the query, use CLIP text-to-image search over the new semantic context!
                        if image_path and q == current_queries[0]:
                            vr = self.retrieve_visual_by_image(image_path)
                        else:
                            vr = self.retrieve_visual(q, k=4)
                        for v in vr:
                            v_results_dict[getattr(v, "id", str(v))] = v
                        visual_report += (
                            f"Visual Docs cho '{q}': {len(vr)} frames found.\n"
                        )

                # Update the main results list with the deduplicated values
                knowledge_results = list(k_results_dict.values())
                visual_results = list(v_results_dict.values())

                thoughts.append(
                    f" Tổng hợp vòng {iteration + 1}: {len(knowledge_results)} văn bản, {len(visual_results)} khung hình."
                )

                #  Verification Agent (Gemma Mode - NO TOOLS/no system_instruction) 
                thoughts.append(
                    f" Verifier đang kiểm chứng dữ liệu (Vòng {iteration + 1}/{max_iterations})..."
                )
                verify_prompt = (
                    "System: Bạn là Verifier Agent. Nhiệm vụ của bạn là đánh giá xem Báo cáo Lore và Báo cáo Visual có ĐỦ thông tin để trả lời Câu hỏi gốc hay không.\n"
                    "Nếu ĐỦ, trả lời chính xác từ: 'SUFFICIENT'.\n"
                    "Nếu THIẾU thông tin cốt lõi, hãy MỞ RỘNG VÀ TÁCH câu hỏi thành 2-3 TỪ KHÓA/CÂU TRUY VẤN MỚI, HOÀN TOÀN KHÁC NHAU để hệ thống tìm kiếm đa chiều.\n"
                    "Trả lời theo cú pháp: 'INSUFFICIENT: [Query 1] | [Query 2]'. Ví dụ: 'INSUFFICIENT: rose titanic | door sinking scene'. KHÔNG CẦN GIẢI THÍCH GÌ THÊM.\n\n"
                    f"Câu hỏi gốc: {query}\n\nBáo cáo Lore:\n{lore_report}\n\nBáo cáo Visual:\n{visual_report}\n\n"
                    "User: Đánh giá của bạn là gì?"
                )

                try:
                    ver_res = self._llm_client.models.generate_content(
                        model=self.model_id,
                        contents=verify_prompt,
                    )
                    ver_text = ver_res.text.strip()
                    if (
                        "SUFFICIENT" in ver_text.upper()
                        and "INSUFFICIENT" not in ver_text.upper()
                    ):
                        thoughts.append(" Verifier: ĐÃ ĐỦ DỮ LIỆU ĐỂ TRẢ LỜI.")
                        break
                    else:
                        parts = ver_text.split("INSUFFICIENT:")
                        if len(parts) > 1 and iteration < effective_max_iter - 1:
                            new_queries_raw = parts[1].strip()
                            current_queries = [
                                q.strip()
                                for q in new_queries_raw.split("|")
                                if q.strip()
                            ]
                            if current_queries:
                                thoughts.append(
                                    f" Verifier yêu cầu tìm lại đa luồng với các truy vấn: {current_queries}"
                                )
                                rewrite_count += 1
                            else:
                                thoughts.append(
                                    "️ Verifier: Parsing mảng query thất bại, dừng vòng lặp."
                                )
                                break
                        else:
                            thoughts.append(
                                "️ Verifier: Không đủ dữ liệu nhưng đã hết lượt tìm."
                            )
                            break
                except Exception as e:
                    thoughts.append(f"️ Lỗi Verifier: {e}. Bỏ qua kiểm chứng.")
                    break

        #  Step 4: Tool-Calling JudgeAgent (Kimi K2 via Groq) 
        thoughts.append(
            "️ JudgeAgent (Kimi K2 + tools) đang tổng hợp và tự gọi công cụ..."
        )

        #  Pre-compute temporal_grounding from best FAISS match 
        best_start_time = ""
        best_end_time = ""
        if visual_results:
            # Pick the highest-scoring match belonging to identified_movie
            for v in sorted(
                visual_results, key=lambda x: getattr(x, "score", 0), reverse=True
            ):
                v_meta = getattr(v, "metadata", {})
                v_mid = getattr(v, "movie_id", v_meta.get("movie_id", ""))
                st = v_meta.get("start_time", "")
                et = v_meta.get("end_time", "")
                if st and et:
                    if not identified_movie or v_mid == identified_movie:
                        best_start_time = st
                        best_end_time = et
                        break

        # Build temporal grounding line for JudgeAgent
        temporal_hint = ""
        if best_start_time and best_end_time:
            temporal_hint = (
                f"\n️ TEMPORAL GROUNDING (từ FAISS metadata — CHÍNH XÁC, dùng nguyên giá trị):\n"
                f'```json\n{{\n  "temporal_grounding": {{\n    "start_time": "{best_start_time}",\n    "end_time": "{best_end_time}"\n  }}\n}}\n```\n'
            )

        #  Build evidence context for JudgeAgent 
        lore_context = "\n".join(
            [
                f"[{i + 1}] {getattr(k, 'text', str(k))[:300]}"
                for i, k in enumerate(knowledge_results)
            ]
        )

        # Load movie metadata for high-confidence visual matches
        import json as _json
        from pathlib import Path as _Path

        _META_DIRS = [
            _Path("data/movienet_subset/meta"),
            _Path("data/unified_dataset/meta"),
            _Path("movie_data_subset_20/meta"),
        ]

        _CHUNK_DIR = _Path("data/temporal_chunks")

        def _load_movie_meta(movie_id: str) -> dict:
            for d in _META_DIRS:
                p = d / f"{movie_id}.json"
                if p.exists():
                    try:
                        return _json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            return {}

        def _load_temporal_chunk(movie_id: str, shot_id: int) -> dict:
            if not movie_id or shot_id is None:
                return {}
            chunk_file = _CHUNK_DIR / f"{movie_id}_chunks.json"
            if not chunk_file.exists():
                return {}
            try:
                chunks = _json.loads(chunk_file.read_text(encoding="utf-8"))
                for chunk in chunks:
                    if (
                        chunk.get("shot_start", 0)
                        <= shot_id
                        <= chunk.get("shot_end", float("inf"))
                    ):
                        return chunk
            except Exception as e:
                thoughts.append(f"️ Failed to parse {chunk_file.name}: {e}")
            return {}

        # Build enriched visual context with movie metadata and full 5-layer chunks
        visual_context_lines = []
        for i, v in enumerate(visual_results):
            mid = getattr(v, "movie_id", "")
            meta = getattr(v, "metadata", {})

            # Extract basic FAISS info
            shot_str = meta.get("shot_id", "")

            # Attempt to parse shot ID to fetch full chunk data
            shot_id_num = None
            import re

            if shot_str:
                shot_match = re.search(r"shot[_-]?(\d+)", str(shot_str), re.IGNORECASE)
                if shot_match:
                    shot_id_num = int(shot_match.group(1))
                elif isinstance(shot_str, int):
                    shot_id_num = shot_str
                elif str(shot_str).isdigit():
                    shot_id_num = int(shot_str)

            # Check if we have the rich chunk info from temporal_chunks
            chunk_data = {}
            if mid and shot_id_num is not None:
                chunk_data = _load_temporal_chunk(mid, shot_id_num)

            # If we found chunk data, OVERRIDE the FAISS metadata with the rich 5-layer chunk
            if chunk_data:
                time_str = ""
                start_time = chunk_data.get("start_time", "")
                end_time = chunk_data.get("end_time", "")

                chunk_desc = chunk_data.get("description", "")
                chunk_characters = chunk_data.get("characters", [])
                chunk_dialogue = chunk_data.get("dialogue_text", "")
                chunk_situation = chunk_data.get("situation", "")
                chunk_cast = chunk_data.get("cast_in_scene", [])
                timestamp_source = chunk_data.get(
                    "timestamp_source", "annotation_frame"
                )
            else:
                # Fallback to FAISS pure metadata if chunks are missing (unlikely if built correctly)
                time_str = meta.get("time", "")
                start_time = meta.get("start_time", "")
                end_time = meta.get("end_time", "")

                chunk_desc = meta.get("description", "")
                chunk_characters = meta.get("characters", [])
                chunk_dialogue = meta.get("dialogue_text", "")
                chunk_situation = meta.get("situation", "")
                chunk_cast = meta.get("cast_in_scene", [])
                timestamp_source = meta.get("timestamp_source", "")

            # Fallback to scene_context if direct time is missing from FAISS and Chunk
            if not start_time and not end_time and not time_str:

                def _fmt(sec):
                    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                scene_ctx = meta.get("scene_context", {})
                t_start = scene_ctx.get("scene_timestamp", 0)
                t_end = scene_ctx.get("scene_timestamp_end", 0)

                if t_start or t_end:
                    start_time = _fmt(t_start)
                    t_end = t_end if t_end else t_start + 15
                    end_time = _fmt(t_end)
                elif shot_str:
                    import re

                    shot_match = re.search(
                        r"shot[_-]?(\d+)", str(shot_str), re.IGNORECASE
                    )
                    if shot_match:
                        shot_num = int(shot_match.group(1))
                        t_start = shot_num * 3
                        t_end = t_start + 15
                        start_time = _fmt(t_start)
                        end_time = _fmt(t_end)

            # Format time display
            if start_time and end_time:
                ts_label = (
                    "Exact" if timestamp_source == "annotation_frame" else "Approx"
                )
                time_disp = f"{start_time} → {end_time} [{ts_label}]"
            elif time_str:
                time_disp = time_str
            else:
                time_disp = "N/A"

            score = getattr(v, "score", 0.0)
            title_hint = meta.get("title", mid)
            line = f"\n"
            line += f"  Temporal Chunk (shot_id: {shot_str})                       \n"
            line += f"                                                             \n"
            line += (
                f"   Layer 1 — Temporal Anchor:                               \n"
            )
            line += (
                f"     movie: {title_hint} [{mid}]                              \n"
            )
            line += f"     time: {time_disp}                                      \n"
            line += f"                                                             \n"

            if chunk_situation or chunk_desc:
                line += f"   Layer 2 — Semantic Description:                          \n"
                if chunk_situation:
                    line += f"     situation: {chunk_situation}                               \n"
                if chunk_desc:
                    line += f"     description: {chunk_desc[:150]}...                         \n"
                line += (
                    f"                                                             \n"
                )

            if chunk_dialogue:
                line += (
                    f"  ️ Layer 3 — Dialogue (from SRT):                           \n"
                )
                line += f'     dialogue_text: "{chunk_dialogue[:100]}..."               \n'
                line += (
                    f"                                                             \n"
                )

            if chunk_characters or chunk_cast:
                line += f"   Layer 4 — Movie Metadata:                                \n"
                if chunk_characters:
                    line += f"     characters: {', '.join(chunk_characters[:5])}              \n"
                if chunk_cast:
                    cast_str = ", ".join(
                        f"{c['actor']} → {c['character']}" for c in chunk_cast[:3]
                    )
                    line += f"     cast_in_scene: {cast_str}                               \n"
                line += (
                    f"                                                             \n"
                )

            line += f""

            if score > 0.70 and mid and mid not in movie_meta_cache:
                movie_meta_cache[mid] = _load_movie_meta(mid)

            visual_context_lines.append(line)
        visual_context = "\n".join(visual_context_lines)

        # Build movie metadata block
        movie_meta_block = ""
        for mid, mmeta in movie_meta_cache.items():
            if mmeta:
                title = mmeta.get("title", mid)
                genres = ", ".join(mmeta.get("genres", []))
                cast = mmeta.get("cast", [])[:8]
                cast_str = ", ".join(
                    f"{c['name']} as {c.get('character', '?')}"
                    for c in cast
                    if c.get("name")
                )
                movie_meta_block += (
                    f"\n {title} [{mid}] | {genres}\n   Cast: {cast_str}\n"
                )

        judge_system = """
        ROLE
        Bạn là một cinephile và nhà phê bình điện ảnh. Bạn hiểu sâu về phim, diễn xuất,
        bối cảnh và storytelling.

        GROUNDING RULES (QUAN TRỌNG NHẤT)
        1. Thông tin trong các khối:
         KẾT QUẢ VISUAL
         THÔNG TIN PHIM
        được xem là NGUỒN SỰ THẬT.

        2. Trong  KẾT QUẢ VISUAL có 5 tầng metadata:
        - Movie
        - Time
        - Characters
        - Scene / Situation
        - Dialogue

        3. Bạn PHẢI bám theo metadata này.
        Nếu metadata nói đây là phim A, bạn KHÔNG được suy đoán thành phim B.

        4. Nếu có mâu thuẫn giữa kiến thức cá nhân và metadata,
        metadata LUÔN đúng.

        REASONING
        Khi trả lời:
        1. Xác định phim.
        2. Xác định cảnh / thời điểm.
        3. Xác định nhân vật xuất hiện.
        4. Diễn giải ý nghĩa cảnh hoặc bối cảnh câu thoại.

        Chỉ sử dụng thông tin có thể suy ra từ metadata hoặc tools.

        STYLE
        - Tự nhiên, như một cinephile đang kể về cảnh phim.
        - Có thể thêm 1 chi tiết điện ảnh nhỏ (diễn xuất, bối cảnh, motif).
        - Tránh lan man hoặc bịa trivia nếu không chắc chắn.

        FORBIDDEN
        KHÔNG BAO GIỜ:
        - Bịa tên phim khác metadata
        - Nhắc tới hệ thống hoặc dữ liệu nội bộ

        Cấm các từ sau:
        CLIP
        cosine
        FAISS
        score
        database
        index
        vector
        embedding
        metadata

        Cấm các câu như:
        "Dựa trên N khung hình"
        "Theo dữ liệu"
        "Hệ thống cho thấy"

        TOOLS
        Nếu metadata chưa đủ để trả lời rõ:
        bạn có thể gọi tool:

        search_knowledge
        search_visual
        search_dialogue
        query_graph

        Chỉ gọi tool khi thực sự thiếu thông tin.

        OUTPUT FORMAT

        Trả lời dạng narrative ngắn gọn.

        Sau phần trả lời,
        nếu hệ thống cung cấp mục ️ temporal_grounding
        bạn PHẢI copy nguyên JSON đó xuống cuối.

        KHÔNG sửa timestamp.
        KHÔNG tạo timestamp mới.

        Ví dụ:

        <narrative answer>

        {
        "temporal_grounding": ...
        }
        """

        #  Build user-facing evidence block 
        judge_user_content = ""

        if movie_meta_block:
            judge_user_content += f" THÔNG TIN PHIM:\n{movie_meta_block}\n\n"

        if vlm_analysis_text:
            judge_user_content += f"{vlm_analysis_text}\n"

        if vlm_conflict_warning:
            judge_user_content += f"{vlm_conflict_warning}\n"

        if visual_context:
            judge_user_content += (
                f" KẾT QUẢ VISUAL (các cảnh khớp nhất):\n{visual_context}\n\n"
            )

        if lore_context:
            judge_user_content += f" KỊCH BẢN & ĐỒ THỊ TRI THỨC:\n{lore_context}\n\n"

        if temporal_hint:
            judge_user_content += f"{temporal_hint}\n"

        judge_user_content += (
            f" CÂU HỎI CỦA NGƯỜI DÙNG: {query}\n\n"
            "Hãy tổng hợp toàn bộ thông tin trên thành câu trả lời tự nhiên, cuốn hút. "
            "Nhớ copy nguyên khối JSON ️ temporal_grounding vào cuối nếu có."
        )

        judge_prompt = f"System: {judge_system}\n\nUser: {judge_user_content}"

        # Build a simple tool executor that calls our Python retrieval methods
        def _tool_executor(tool_name: str, tool_args: dict) -> str:
            k_arg = tool_args.get("k", 5)
            q_arg = tool_args.get("query", query)
            if tool_name == "search_knowledge":
                results = self.retrieve_knowledge(q_arg, k=k_arg)
                thoughts.append(
                    f" Kimi called search_knowledge('{q_arg}') → {len(results)} docs"
                )
                return "\n".join(
                    [
                        f"[{i + 1}] {getattr(r, 'text', str(r))[:300]}"
                        for i, r in enumerate(results)
                    ]
                )
            elif tool_name == "search_visual":
                results = self.retrieve_visual(q_arg, k=k_arg)
                thoughts.append(
                    f" Kimi called search_visual('{q_arg}') → {len(results)} frames"
                )
                # also extend visual_results for keyframe display
                v_dict = {getattr(v, "id", str(v)): v for v in visual_results}
                for r in results:
                    v_dict[getattr(r, "id", str(r))] = r
                visual_results[:] = list(v_dict.values())
                return "\n".join(
                    [
                        f"[{i + 1}] Movie:{getattr(r, 'movie_id', '')} Shot:{getattr(r, 'metadata', {}).get('shot_id', '')} Score:{getattr(r, 'score', 0):.2f}"
                        for i, r in enumerate(results)
                    ]
                )
            elif tool_name == "search_dialogue":
                di = getattr(self, "dialogue_indexer", None)
                if di:
                    results = di.search(q_arg, k=k_arg)
                    thoughts.append(
                        f" Kimi called search_dialogue('{q_arg}') → {len(results)} lines"
                    )
                    return "\n".join(
                        [
                            f"[{i + 1}] {r.get('text', '')[:200]}"
                            for i, r in enumerate(results)
                        ]
                    )
                return "[dialogue index not available]"
            elif tool_name == "query_graph":
                if hasattr(self, "query_graph"):
                    cypher = str(tool_args.get("query", q_arg) or "")
                    thoughts.append(f" Kimi called query_graph('{cypher}')")
                    return str(self.query_graph(cypher))
                return "[Graph Database not connected]"
            return f"[Unknown tool: {tool_name}]"

        try:
            judge_res = self._llm_client.generate_with_tools(
                prompt=judge_prompt,
                tool_executor=_tool_executor,
                max_tool_rounds=5,
            )
            answer = judge_res.text.strip()
        except Exception as e:
            thoughts.append(
                f"️ Tool-calling JudgeAgent failed: {e}. Trying plain generate."
            )
            try:
                plain_res = self._llm_client.models.generate_content(
                    model="kimi", contents=judge_prompt
                )
                answer = plain_res.text.strip()
            except Exception as e2:
                answer = f"Lỗi JudgeAgent: {e2}. Vui lòng thử lại sau."
                thoughts.append(f" Lỗi JudgeAgent: {e2}")

        #  Step 5: Collect keyframe paths 
        keyframe_paths = []
        for r in visual_results[:6]:
            img_path = getattr(r, "path", "")
            meta_dict = getattr(r, "metadata", {})
            if not isinstance(meta_dict, dict):
                meta_dict = {}

            if not img_path:
                img_path = meta_dict.get("path", "")

            if img_path and os.path.exists(img_path):
                shot = meta_dict.get("shot_id", "frame")
                caption = f"{getattr(r, 'movie_id', 'Unknown')} | {shot} | {getattr(r, 'score', 0.0):.2f}"
                keyframe_paths.append((img_path, caption))

        return {
            "answer": answer,
            "intent": intent.value if hasattr(intent, "value") else str(intent),
            "visual_results": visual_results,
            "knowledge_results": knowledge_results,
            "keyframe_paths": keyframe_paths,
            "temporal_grounding": {
                "start_time": best_start_time,
                "end_time": best_end_time,
            }
            if best_start_time and best_end_time
            else None,
            "thoughts": thoughts,
        }
