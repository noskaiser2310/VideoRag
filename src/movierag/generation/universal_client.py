"""
Universal LLM Client wrapper for MovieRAG.

Priority chain:
1. Groq → moonshotai/kimi-k2-instruct (fastest, free-tier-friendly)
"""

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

_KIMI_MODEL = "moonshotai/kimi-k2-instruct"


class UniversalLLMClient:
    """
    Unified client that routes requests using: Groq(Kimi).
    """

    def __init__(self, model_id: str = _KIMI_MODEL):
        self.model_id = model_id

        # Environment variables like GROQ_API_KEY are expected to be loaded via dotenv

        self._groq_client = self._init_groq()

        # Mock google.genai API namespace
        _self = self

        class ModelsMock:
            def generate_content(self, model: str, contents: Any, **kwargs) -> Any:
                return _self.generate_content(model, contents, **kwargs)

        self.models = ModelsMock()

    #  Init helpers 

    def _init_groq(self):
        try:
            from groq import Groq

            logger.info(f"Groq client initialized — primary model: {_KIMI_MODEL}")
            return Groq(api_key=os.environ["GROQ_API_KEY"])
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")
            return None

    #  Response wrapper 

    class GenerateContentResponse:
        def __init__(self, text: str):
            self.text = text

    #  Core generate method 

    def generate_content(
        self, model: str, contents: Any, **kwargs
    ) -> GenerateContentResponse:
        """Attempt Groq(Kimi) for generation."""

        # --- Build text prompt ---
        prompt = ""
        if isinstance(contents, str):
            prompt = contents
        elif isinstance(contents, list):
            prompt = "\n".join(c for c in contents if isinstance(c, str))

        messages = self._build_messages(prompt)

        #  Try Kimi via Groq 
        if self._groq_client:
            models_to_try = [
                (
                    "moonshotai/kimi-k2-instruct",
                    {
                        "temperature": kwargs.get("temperature", 0.6),
                        "max_completion_tokens": 16384,
                    },
                ),
                (
                    "qwen/qwen3-32b",
                    {
                        "temperature": kwargs.get("temperature", 0.6),
                        "max_completion_tokens": 8000,
                        "top_p": 0.95,
                    },
                ),
                (
                    "openai/gpt-oss-120b",
                    {
                        "temperature": 1.0,
                        "max_completion_tokens": 8192,
                        "top_p": 1,
                        "extra_body": {"reasoning_effort": "high"},
                    },
                ),
            ]

            completion = None
            used_model = ""
            import time

            MAX_RETRIES = 5
            for attempt in range(MAX_RETRIES):
                # Swap models per attempt if possible
                model_idx = attempt % len(models_to_try)
                model_name, model_kwargs = models_to_try[model_idx]

                try:
                    logger.info(
                        f"Trying Groq model: {model_name} (Attempt {attempt + 1}/{MAX_RETRIES})..."
                    )
                    kwargs_copy = model_kwargs.copy()
                    extra = kwargs_copy.pop("extra_body", None)
                    call_args = {
                        "model": model_name,
                        "messages": messages,
                        "stream": True,
                        "stop": None,
                        **kwargs_copy,
                    }
                    if extra:
                        call_args["extra_body"] = extra

                    completion = self._groq_client.chat.completions.create(**call_args)
                    used_model = model_name
                    break  # Success
                except Exception as e:
                    logger.error(f"Groq call failed for {model_name}: {e}")
                    time.sleep(1 + attempt)  # Exponential backoff before retry

            if completion:
                try:
                    full_text = ""
                    for chunk in completion:
                        full_text += chunk.choices[0].delta.content or ""

                    logger.info(
                        f"Groq ({used_model}) responded ({len(full_text)} chars)"
                    )
                    return self.GenerateContentResponse(text=full_text)
                except Exception as e:
                    logger.error(f"Error reading Groq stream: {e}")
                    raise e

            # If we reach here, all Groq models failed 5 times
            raise RuntimeError("All Groq retry attempts failed.")

        raise RuntimeError("No Groq LLM client available. Set GROQ_API_KEY.")

    def generate_vision_content(self, prompt: str, image_path: str, **kwargs) -> str:
        """Use VLM (Llama 3.2 Vision on Groq, or Gemini fallback) to analyze an image."""
        import base64
        import mimetypes

        def encode_image(img_path):
            with open(img_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        base64_image = encode_image(image_path)

        #  Try Groq Vision first 
        if self._groq_client:
            try:
                completion = self._groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=kwargs.get("temperature", 0.6),
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                )
                logger.info("Groq/Vision responded.")
                return completion.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Groq/Vision analysis failed: {e}")
                return f"[Lỗi VLM Groq: {e}]"

        return "[No VLM available]"

    #  Helper 

    @staticmethod
    def _build_messages(prompt: str) -> list:
        """Convert plain prompt string into OpenAI-style messages list."""
        if "System:" in prompt and "User:" in prompt:
            parts = prompt.split("User:", 1)
            sys_part = parts[0].replace("System:", "").strip()
            user_part = parts[1].strip()
            return [
                {"role": "system", "content": sys_part},
                {"role": "user", "content": user_part},
            ]
        return [{"role": "user", "content": prompt}]

    #  Tool Calling 

    # Standard MovieRAG tool schemas (OpenAI function-calling format)
    MOVIERAG_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the MovieRAG knowledge database (scripts, subtitles, metadata) for text information about movies, actors, plot, or dialogue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in natural language",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_visual",
                "description": "Search the visual FAISS index to find relevant keyframes/scenes by text description. Use when the user asks about appearances, scenes, or visual details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Visual scene description",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max frames (default 6)",
                            "default": 6,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_dialogue",
                "description": "Search subtitles/dialogue index for specific quotes or spoken lines in movies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Quote or dialogue fragment",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Max results (default 4)",
                            "default": 4,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    def generate_with_tools(
        self,
        prompt: str,
        tool_executor,  # callable: (tool_name, tool_args) → str
        tools: list | None = None,
        max_tool_rounds: int = 5,
        **kwargs,
    ) -> GenerateContentResponse:
        """
        Agentic tool-calling loop with Kimi/Groq.

        - Sends the prompt + tool schemas to Kimi.
        - If Kimi returns a tool_call, executes the corresponding Python function via `tool_executor`.
        - Feeds results back to the model until Kimi gives a final text response.
        - Falls back to plain `generate_content` if Groq is unavailable.

        Args:
            prompt: User + context prompt string.
            tool_executor: Callable(tool_name: str, tool_args: dict) → str (tool result as text).
            tools: Optional override of tool schemas. Defaults to MOVIERAG_TOOLS.
            max_tool_rounds: Safety cap to avoid infinite loops.
        """
        if not self._groq_client:
            logger.warning(
                "Tool calling requires Groq client. Falling back to plain generate."
            )
            return self.generate_content("kimi", prompt, **kwargs)

        import json

        tools = tools or self.MOVIERAG_TOOLS
        messages = self._build_messages(prompt)

        import time

        for _round in range(max_tool_rounds):
            resp = None

            models_to_try = [
                (
                    "moonshotai/kimi-k2-instruct",
                    {
                        "temperature": kwargs.get("temperature", 0.6),
                        "max_completion_tokens": 4096,
                    },
                ),
                (
                    "qwen/qwen3-32b",
                    {
                        "temperature": kwargs.get("temperature", 0.6),
                        "max_completion_tokens": 8000,
                        "top_p": 0.95,
                    },
                ),
                (
                    "openai/gpt-oss-120b",
                    {
                        "temperature": 1.0,
                        "max_completion_tokens": 4096,
                        "top_p": 1,
                        "extra_body": {"reasoning_effort": "high"},
                    },
                ),
            ]

            MAX_RETRIES = 5
            for attempt in range(MAX_RETRIES):
                # Swap models per attempt if possible
                model_idx = attempt % len(models_to_try)
                model_name, model_kwargs = models_to_try[model_idx]

                try:
                    logger.info(
                        f"Tool-calling Groq request with {model_name} (Attempt {attempt + 1}/{MAX_RETRIES})..."
                    )
                    kwargs_copy = model_kwargs.copy()
                    extra = kwargs_copy.pop("extra_body", None)
                    call_args = {
                        "model": model_name,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        **kwargs_copy,
                    }
                    if extra:
                        call_args["extra_body"] = extra

                    resp = self._groq_client.chat.completions.create(**call_args)
                    break  # Success
                except Exception as e:
                    logger.warning(
                        f"Tool-calling Groq request failed for {model_name}: {e}."
                    )
                    time.sleep(1 + attempt)

            if not resp:
                logger.warning("All Groq models failed. Reverting to base generation.")
                return self.generate_content("kimi", prompt, **kwargs)

            choice = resp.choices[0]
            finish = choice.finish_reason

            #  Model returned a final text answer 
            if finish == "stop" or not choice.message.tool_calls:
                text = choice.message.content or ""
                logger.info(
                    f"Kimi responded after {_round} tool round(s) ({len(text)} chars)"
                )
                return self.GenerateContentResponse(text=text)

            #  Model requested one or more tool calls 
            # Safely append the assistant's message, stripping unsupported fields like 'reasoning'
            m_dict = choice.message.model_dump(exclude_unset=True)
            if "reasoning" in m_dict:
                del m_dict["reasoning"]
            messages.append(m_dict)

            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except Exception:
                    fn_args = {}

                logger.info(f"Kimi calls tool: {fn_name}({fn_args})")
                try:
                    tool_result = tool_executor(fn_name, fn_args)
                except Exception as ex:
                    tool_result = f"[Tool error: {ex}]"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_result),
                    }
                )

        logger.warning(
            "Tool calling hit max_tool_rounds limit — returning last content."
        )
        last_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        return self.GenerateContentResponse(text=last_content)
