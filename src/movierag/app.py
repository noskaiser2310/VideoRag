"""
MovieRAG Complete Web Application
=================================
Premium dark-mode interface with chat-embedded upload and compact evidence panel.
"""

import logging
import json
import re

try:
  import gradio as gr

  GRADIO_AVAILABLE = True
except ImportError:
  GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  background: #0b0d14 !important;
  font-family: 'Inter', 'Segoe UI', sans-serif !important;
  color: #dde1f0 !important;
  margin: 0 !important;
}

/* App wrapper */
#app-root { max-width: 1200px; margin: 0 auto; padding: 16px; }

/* Header */
.mr-header {
  background: linear-gradient(135deg, #13172e 0%, #0d1028 100%);
  border: 1px solid rgba(99,102,241,.25);
  border-radius: 14px;
  padding: 18px 24px 14px;
  margin-bottom: 14px;
}
.mr-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; }
.mr-header p { margin: 2px 0 0; font-size: .78rem; color: #7c85c0; }
.grad { background: linear-gradient(90deg,#818cf8,#a78bfa,#f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* Main layout */
.main-col { display: flex; flex-direction: column; gap: 10px; }

/* Chatbot */
#chatbot {
  background: #0f1219 !important;
  border: 1px solid rgba(99,102,241,.2) !important;
  border-radius: 12px !important;
  min-height: 440px !important;
}
/* chat bubbles */
.message.user .message-bubble-border {
  background: linear-gradient(135deg,#4338ca,#6366f1) !important;
  border-radius: 18px 18px 4px 18px !important;
}
.message.bot .message-bubble-border {
  background: #161b2e !important;
  border: 1px solid rgba(99,102,241,.18) !important;
  border-radius: 18px 18px 18px 4px !important;
}

/* Chat input bar */
#chat-bar {
  background: #161b2e !important;
  border: 1.5px solid rgba(99,102,241,.3) !important;
  border-radius: 14px !important;
  padding: 6px 8px !important;
  display: flex !important;
  align-items: flex-end !important;
  gap: 6px !important;
  transition: border-color .2s;
}
#chat-bar:focus-within { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important; }
#chat-txt textarea {
  background: transparent !important;
  border: none !important;
  color: #e2e8f0 !important;
  font-size: .92rem !important;
  resize: none !important;
  padding: 6px 4px !important;
}

/* upload mini-buttons inside bar */
.upload-btn {
  background: rgba(99,102,241,.12) !important;
  border: 1px solid rgba(99,102,241,.25) !important;
  border-radius: 9px !important;
  color: #a5b4fc !important;
  font-size: .75rem !important;
  padding: 5px 10px !important;
  cursor: pointer;
  white-space: nowrap;
  transition: background .15s;
}
.upload-btn:hover { background: rgba(99,102,241,.25) !important; }

/* hidden large upload areas – only shown when toggled */
#img-drop, #vid-drop {
  border: 1.5px dashed rgba(99,102,241,.3) !important;
  border-radius: 10px !important;
  background: #11152a !important;
  padding: 8px !important;
}

#send-btn {
  background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
  border: none !important; border-radius: 10px !important;
  font-weight: 650 !important; font-size: .9rem !important;
  padding: 8px 18px !important; white-space: nowrap;
  transition: transform .15s, box-shadow .15s;
}
#send-btn:hover { transform: scale(1.04); box-shadow: 0 4px 20px rgba(99,102,241,.4) !important; }

/* Status strip */
#status-txt {
  background: transparent !important;
  border: none !important;
  border-top: 1px solid rgba(99,102,241,.1) !important;
  border-radius: 0 !important;
  font-size: .72rem !important;
  color: #6b7bbf !important;
  padding: 2px 4px !important;
}
#status-txt textarea { color: #6b7bbf !important; font-size: .72rem !important; padding: 0 !important; }

/* Evidence Accordion */
.evidence-accordion {
  background: #0f1219 !important;
  border: 1px solid rgba(99,102,241,.2) !important;
  border-radius: 12px !important;
  overflow: hidden;
}
.evidence-accordion .label-wrap {
  background: linear-gradient(90deg,rgba(99,102,241,.12),transparent) !important;
  padding: 8px 14px !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
  color: #a5b4fc !important;
  letter-spacing: .5px;
}

/* Gallery compact */
#gallery {
  background: #0f1219 !important;
  border-radius: 0 0 10px 10px !important;
}
.gallery-item img {
  border-radius: 7px !important;
  cursor: zoom-in !important;
  transition: transform .2s, box-shadow .2s !important;
}
.gallery-item img:hover { transform: scale(1.06); box-shadow: 0 6px 24px rgba(99,102,241,.35) !important; }

/* Video compact */
#vid-player { border-radius: 0 0 10px 10px !important; }
#vid-player video { border-radius: 8px !important; }

/* Tabs */
.tab-nav { background: #0f1219 !important; border-bottom: 1px solid rgba(99,102,241,.18) !important; }
.tab-nav button { color: #7c85c0 !important; font-size: .78rem !important; padding: 7px 14px !important; }
.tab-nav button.selected { color: #a5b4fc !important; border-bottom: 2px solid #6366f1 !important; }

/* Clear btn */
#clear-btn {
  background: rgba(239,68,68,.08) !important;
  border: 1px solid rgba(239,68,68,.2) !important;
  color: #fca5a5 !important; border-radius: 8px !important;
  font-size: .75rem !important;
}
#clear-btn:hover { background: rgba(239,68,68,.18) !important; }

/* Media Accordion */
#media-acc {
  background: transparent !important;
  border: 1px dashed rgba(99,102,241,.25) !important;
  border-radius: 8px !important;
  margin-top: 8px !important;
}
#media-acc > .label-wrap {
  color: #a5b4fc !important;
  font-size: 0.85rem !important;
}

/* Examples */
.examples .example { background: #161b2e !important; border: 1px solid rgba(99,102,241,.18) !important; border-radius: 8px !important; font-size:.8rem !important; }
.examples .example:hover { border-color: #6366f1 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: #2d3455; border-radius: 99px; }
"""


def create_integrated_app(pipeline=None):
  if not GRADIO_AVAILABLE:
    raise ImportError("Gradio not installed.")

  # Backend 
  def respond(user_message, image_input, video_input, chat_history):
    if not user_message and image_input is None and video_input is None:
      return chat_history, [], None, "️ Ready"

      chat_history = chat_history + [
        {"role": "user", "content": user_message or "Media"},
        {"role": "assistant", "content": "Pipeline chưa khởi tạo."},
      ]
      return chat_history, [], None, "No pipeline"

    result = pipeline.respond(
      query=user_message or "",
      image_path=image_input,
      video_path=video_input,
      history=chat_history,
    )

    intent = result["intent"]
    answer = result["answer"]
    thoughts = result["thoughts"]
    knowledge_results = result["knowledge_results"]
    visual_results = result["visual_results"]

    # Parse temporal grounding 
    temporal_info = None
    clean_answer = answer

    # Look for temporal_grounding key within any curly braces block
    m = re.search(r'(\{[\s\S]*?"temporal_grounding"[\s\S]*?\})', answer)

    # 1st Priority: Native temporal grounding from pipeline response
    if "temporal_grounding" in result and result["temporal_grounding"]:
      temporal_info = result["temporal_grounding"]
      if m:
        # Still scrub the JSON out of the chat message if present
        json_str = m.group(1)
        full_match = re.search(
          rf"```json\s*{re.escape(json_str)}\s*```", answer
        )
        if full_match:
          clean_answer = answer.replace(full_match.group(0), "").strip()
        else:
          clean_answer = answer.replace(json_str, "").strip()

    # 2nd Priority: Fallback to old regex parsing if pipeline didn't provide it
    elif m:
      json_str = m.group(1)
      try:
        parsed = json.loads(json_str)
        if "temporal_grounding" in parsed:
          temporal_info = parsed["temporal_grounding"]
          full_match = re.search(
            rf"```json\s*{re.escape(json_str)}\s*```", answer
          )
          if full_match:
            clean_answer = answer.replace(full_match.group(0), "").strip()
          else:
            clean_answer = answer.replace(json_str, "").strip()
      except json.JSONDecodeError:
        pass

    # Build message
    bot_msg = ""
    if thoughts:
      t_html = "<br>→ ".join(thoughts)
      bot_msg += f"<details><summary><b>Agent Thoughts</b></summary>\n\n_{t_html}_\n\n</details>\n\n"
    bot_msg += clean_answer + "\n"

    if knowledge_results:
      bot_msg += "\n---\n**Sources:**\n"
      for i, r in enumerate(knowledge_results[:3]):
        try:
          meta = getattr(r, "metadata", r)
          title = (
            meta.get("title", getattr(r, "movie_id", "?"))
            if isinstance(meta, dict)
            else str(r)[:35]
          )
          score = getattr(r, "score", 0.0)
          bot_msg += f"`[{i + 1}]` {title} · {score:.2f}\n"
        except Exception:
          bot_msg += f"`[{i + 1}]` [Ref]\n"

    if visual_results:
      bot_msg += "\n**Matched Frames:**\n"
      for i, r in enumerate(visual_results[:5]):
        try:
          meta = getattr(r, "metadata", r)
          shot = meta.get("shot_id", "") if isinstance(meta, dict) else ""
          score = getattr(r, "score", 0.0)
          mid = getattr(r, "movie_id", "")
          bot_msg += f"`[{i + 1}]` {mid}›{shot} · **{score:.2f}**\n"
        except Exception:
          bot_msg += f"`[{i + 1}]` [Frame]\n"

    # Gallery 
    gallery_images = []
    if intent in ("VISUAL", "MULTIMODAL"):
      gallery_images = result.get("keyframe_paths", [])

    # Clip extraction 
    video_output = None
    if (
      visual_results
      and hasattr(pipeline, "visual_indexer")
      and pipeline.visual_indexer
    ):
      top = visual_results[0]
      mid = getattr(top, "movie_id", "")
      if (
        temporal_info
        and mid
        and hasattr(pipeline.visual_indexer, "extract_clip_at_time")
      ):
        video_output = pipeline.visual_indexer.extract_clip_at_time(
          mid,
          temporal_info.get("start_time", "00:00:00"),
          end_time=temporal_info.get("end_time"),
          duration=15,
        )
      elif mid and intent in ("VISUAL", "MULTIMODAL"):
        fp = getattr(top, "path", "") or top.metadata.get("path", "")
        if fp and hasattr(pipeline.visual_indexer, "extract_video_clip"):
          video_output = pipeline.visual_indexer.extract_video_clip(
            mid, fp, duration=12
          )

    # Status 
    icon = {"VISUAL": "️", "KNOWLEDGE": "", "MULTIMODAL": "", "CHAT": ""}.get(
      intent, ""
    )
    status = f"{icon} {intent} · {len(knowledge_results)} docs · {len(visual_results)} frames"

    disp = user_message if user_message else " [media]"
    chat_history = chat_history + [
      {"role": "user", "content": disp},
      {"role": "assistant", "content": bot_msg},
    ]
    return chat_history, gallery_images, video_output, status

  def clear_all():
    """Clear entire session — no context bleeding between queries."""
    # Reset pipeline's internal chat history if available
    if pipeline and hasattr(pipeline, "_chat_history"):
      pipeline._chat_history.clear()
    return [], [], None, "", None, None, "️ New session started"

  # UI 
  with gr.Blocks(css=CUSTOM_CSS, title="MovieRAG") as app:
    # Header 
    gr.HTML("""
    <div class="mr-header">
     <h1> <span class="grad">MovieRAG</span></h1>
     <p>Agentic VideoRAG · Kimi K2 + CLIP + FAISS · Multi-round Retrieval</p>
    </div>
    """)

    with gr.Row(equal_height=False):
      # LEFT: Chat 
      with gr.Column(scale=7, elem_classes="main-col"):
        chatbot = gr.Chatbot(
          label="",
          elem_id="chatbot",
          height=460,
        )

        # Compact input bar 
        with gr.Group(elem_id="chat-bar"):
          with gr.Row():
            txt = gr.Textbox(
              placeholder="Hỏi về phim... hoặc upload ảnh/video bên dưới",
              show_label=False,
              scale=1,
              container=False,
              elem_id="chat-txt",
              lines=1,
            )
            send = gr.Button(
              "",
              variant="primary",
              scale=0,
              min_width=50,
              elem_id="send-btn",
            )
            new_chat = gr.Button(
              "",
              variant="secondary",
              scale=0,
              min_width=50,
              elem_id="new-chat-btn",
            )

          # Upload row inside Accordion 
          with gr.Accordion(
            " Đính kèm Ảnh / Video", open=False, elem_id="media-acc"
          ):
            with gr.Row():
              img = gr.Image(
                type="filepath",
                label=" Tải Ảnh Lên",
                scale=1,
                height=140,
                elem_id="img-drop",
                show_label=True,
              )
              vid = gr.Video(
                label=" Tải Video Lên",
                scale=1,
                height=140,
                include_audio=False,
                elem_id="vid-drop",
                show_label=True,
              )
            with gr.Row():
              clear = gr.Button(
                "️ Xóa Media Đính Kèm", size="sm", elem_id="clear-btn"
              )

        # Status strip
        status = gr.Textbox(
          value="️ Ready",
          label="",
          interactive=False,
          max_lines=1,
          elem_id="status-txt",
        )

        # Examples 
        gr.Examples(
          label=" Examples",
          examples=[
            ["Ai đóng vai Jack trong Titanic?", None, None],
            ["Tìm cảnh con tàu Titanic chìm", None, None],
            ["Tóm tắt phim Home Alone", None, None],
            ["Tìm cảnh rượt đuổi xe hơi", None, None],
          ],
          inputs=[txt, img, vid],
        )

      # RIGHT: Evidence 
      with gr.Column(scale=3, elem_classes="main-col"):
        gr.HTML(
          '<p style="font-size:.72rem;color:#6b7bbf;text-transform:uppercase;letter-spacing:.8px;margin:2px 0 6px"> Visual Evidence</p>'
        )

        with gr.Tabs():
          with gr.Tab("️ Keyframes"):
            gallery = gr.Gallery(
              label="",
              elem_id="gallery",
              columns=2,
              rows=3,
              height=340,
              object_fit="cover",
              show_label=False,
            )

          with gr.Tab(" Clip"):
            video_player = gr.Video(
              label="",
              elem_id="vid-player",
              height=300,
              interactive=False,
              show_label=False,
            )

    # Events 
    _out = [chatbot, gallery, video_player, status]

    send.click(respond, [txt, img, vid, chatbot], _out).then(
      lambda: "", None, [txt]
    )
    txt.submit(respond, [txt, img, vid, chatbot], _out).then(
      lambda: "", None, [txt]
    )
    clear.click(
      clear_all, None, [chatbot, gallery, video_player, txt, img, vid, status]
    )
    new_chat.click(
      clear_all, None, [chatbot, gallery, video_player, txt, img, vid, status]
    )

  return app
