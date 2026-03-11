"""
LLM Generator Module for MovieRAG.

Handles formatting retrieved context and calling the Google GenAI API
to generate natural language answers seamlessly.
"""

import os
import logging
from typing import List, Dict, Optional, Any

from movierag.indexing.knowledge_indexer import TextSearchResult
from movierag.indexing.visual_indexer import SearchResult as VisualSearchResult

# Define GENAI feature toggle to false to remain API compat if needed elsewhere
GENAI_AVAILABLE = False

from movierag.indexing.knowledge_indexer import TextSearchResult
from movierag.indexing.visual_indexer import SearchResult as VisualSearchResult

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Generates answers using Google GenAI API based on retrieved context.
    """

    def __init__(
        self,
        model_id: str = "gemma-3-27b-it",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM Generator using UniversalLLMClient.

        Args:
            model_id: The model ID to use (e.g., 'moonshotai/kimi-k2-instruct', 'gemma-2-27b-it').
            api_key: Optional API token. If None, looks for GROQ_API_KEY.
        """
        self.model_id = model_id
        self.client = None

        try:
            from movierag.generation.universal_client import UniversalLLMClient

            self.client = UniversalLLMClient(model_id=self.model_id)
            logger.info(f"Initialized LLM Generator with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize UniversalLLMClient: {e}")

    def format_prompt(
        self,
        query: str,
        context_results: List[Any],
        visual_results: List[Any],
        history: List[Dict],
        route: Optional[Any] = None,
    ) -> str:
        """
        Format prompt dynamically based on retrieved context and intent route.
        """
        # Intent handling (supports both old QueryRoute and new QueryIntent)
        intent_value = (
            route.value
            if route and hasattr(route, "value")
            else str(route)
            if route
            else "MULTIMODAL"
        )

        base_prompt = (
            "Bạn là MovieRAG, trợ lý chuyên sâu về phim ảnh có năng lực trích xuất video.\n"
            "Hệ thống truy xuất của chúng tôi ĐÃ TÌM THẤY các thông tin sau từ Database.\n"
            "Nhiệm vụ của bạn là TỔNG HỢP các thông tin này để trả lời câu hỏi.\n"
            "BẮT BUỘC trích dẫn nguồn bằng cách thêm số `[1]`, `[2]` vào cuối câu nếu dùng thông tin đó.\n"
            "Nếu thông tin được cung cấp bên dưới không đủ để trả lời, hãy nói rõ, TUYỆT ĐỐI KHÔNG TỰ BỊA ĐẶT.\n\n"
            " **Yêu cầu BẮT BUỘC về Temporal Grounding (Khoanh vùng thời gian)**:\n"
            "Nếu câu trả lời của bạn có mô tả một cảnh phim cụ thể được lấy từ 'Visual Evidence', bạn PHẢI in ra một block JSON duy nhất ở cuối câu trả lời chứa `start_time` và `end_time` của cảnh đó, ví dụ:\n"
            "```json\n"
            "{\n"
            '  "temporal_grounding": {\n'
            '    "start_time": "00:01:23",\n'
            '    "end_time": "00:01:28"\n'
            "  }\n"
            "}\n"
            "```\n"
            "Chỉ in JSON này nếu bạn chắc chắn về mốc thời gian từ dữ liệu cung cấp. Nếu không, bỏ qua.\n\n"
        )

        #  Route-Specific Instructions 
        if intent_value == "VISUAL":
            base_prompt += (
                " CHÚ Ý [TÌM CẢNH PHIM]: Câu hỏi này yêu cầu tìm kiếm bằng hình ảnh.\n"
                "- Ưu tiên mô tả các kết quả từ phần 'Visual Evidence'.\n"
                "- Phân tích mô tả cảnh, nhân vật, hành động trong các frame đó.\n\n"
            )
        elif intent_value == "KNOWLEDGE":
            base_prompt += (
                " CHÚ Ý [TRA CỨU THÔNG TIN]: Câu hỏi này hỏi về thông tin văn bản (sự kiện, diễn viên, đạo diễn).\n"
                "- Tập trung vào phần 'Knowledge Evidence'.\n"
                "- Bỏ qua hình ảnh nếu không cần thiết.\n\n"
            )
        elif intent_value == "DIALOG":
            base_prompt += (
                " CHÚ Ý [TÌM LỜI THOẠI]: Câu hỏi này xoay quanh lời thoại phim.\n"
                "- Kiểm tra kỹ subtitle trong 'Knowledge Evidence' khớp với câu hỏi.\n\n"
            )
        elif intent_value == "MULTIMODAL" or intent_value in ["REASONING", "TEMPORAL"]:
            base_prompt += (
                " CHÚ Ý [SUY LUẬN ĐA PHƯƠNG THỨC]: Câu hỏi này cần kết hợp cả thông tin text và hình ảnh.\n"
                "- Tìm sự kết nối giữa 'Knowledge Evidence' và 'Visual Evidence'.\n"
                "- Đặc biệt chú ý đến timestamp/thời gian để khớp cảnh với sự kiện.\n\n"
            )
        else:  # Default/Factual
            base_prompt += " CHÚ Ý [TRẢ LỜI THỰC TẾ]: Cung cấp câu trả lời ngắn gọn, trực tiếp dựa trên các kết quả truy xuất.\n\n"

        #  Context Formatting 
        context_str = ""

        # Add Knowledge Context (including MovieGraph)
        if context_results:
            context_str += (
                "--- KNOWLEDGE EVIDENCE (từ cơ sở dữ liệu phim & Knowledge Graph) ---\n"
            )
            for i, result in enumerate(context_results):
                content = result.text.strip()[:500]
                movie = result.movie_id
                category = result.metadata.get("category", "info")
                title = result.metadata.get("title", movie)

                # Highlight MovieGraph explicit structures
                if category == "moviegraph":
                    context_str += f"[GraphRAG Context {i + 1} | Phim: {title} | Loại: Quan hệ/Hành động]\n{content}\n\n"
                else:
                    context_str += f"[Doc {i + 1} | Phim: {title} ({movie}) | Loại: {category}]\n{content}\n\n"

        # Add Visual Context
        if visual_results:
            context_str += "--- VISUAL EVIDENCE (các khung hình hệ thống của chúng tôi tìm thấy) ---\n"
            for i, r in enumerate(visual_results):
                movie = r.movie_id
                shot_id = r.metadata.get("shot_id", "unknown")
                score = r.score
                context_str += f"[Visual Match {i + 1} | Phim: {movie} | Cảnh: {shot_id} | Độ tin cậy: {score:.3f}]\n"

        if not context_str:
            context_str = (
                "No relevant results were found in our database for this query.\n"
            )

        # Add History
        history_str = ""
        if history:
            history_str = "--- CONVERSATION HISTORY ---\n"
            for msg in history[-5:]:
                if isinstance(msg, dict):
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_str += f"{role}: {msg.get('content', '')}\n"
                elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                    history_str += f"User: {msg[0]}\nAssistant: {msg[1]}\n"
            history_str += "\n"

        base_instructions = (
            "You are MovieRAG, a movie expert AI assistant. "
            "Our retrieval system has ALREADY searched the database and found the results below. "
            "Your job is to SYNTHESIZE these results into a helpful, natural language answer. "
            "NEVER say you cannot search or find information — the search is already done for you.\n"
        )

        if intent_value == "MULTIMODAL" or intent_value in ["REASONING"]:
            instruction = "Analyze the evidence deeply: explain character motives, causality, and connections."
        elif intent_value == "TEMPORAL":
            instruction = "Focus on WHEN events happen, the chronological sequence, and timestamps."
        elif intent_value == "DIALOG":
            instruction = (
                "Focus on dialog and quotes. Identify who said what and the context."
            )
        else:  # FACTUAL / KNOWLEDGE / VISUAL
            instruction = (
                "Provide a concise, direct answer based on the retrieved results."
            )

        prompt = f"""{base_instructions}
Task: {instruction}

Rules:
- Use the RETRIEVED results below to form your answer. They are real search results.
- Reference specific movies, shots, and details from the context.
- If a user uploads an image and visual matches exist, tell them which movie and scene was identified.
- Answer in the same language as the user's question.

{history_str}
{context_str}

User's Question: {query}
Answer:"""

        return prompt

    def generate_answer(
        self,
        query: str,
        context_results: List[TextSearchResult],
        visual_results: Optional[List[VisualSearchResult]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        route: Any = None,
    ) -> str:
        """
        Generate an answer using the LLM.
        """
        if not self.client:
            return (
                "LLM generation is unavailable. Please ensure `UniversalLLMClient` initialized properly "
                "and a valid API key is provided (GROQ_API_KEY environment variable)."
            )

        prompt = self.format_prompt(
            query, context_results, visual_results, history, route=route
        )

        try:
            # We construct a simple prompt string for the generation.
            # If visual results are needed to be truly "seen" by the LLM, we should be calling generate_vision_content
            # from UniversalLLMClient, but for standard answer generation, we just use the text-based evidence.
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return f"Error generating answer: {str(e)}"
