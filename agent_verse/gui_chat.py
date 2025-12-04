import time
import html
import gradio as gr
from coordinator import execute_step
from plan import plan_task
from typing import List, Dict

# ---------- Config ----------
CHUNK_SIZE = 80


STREAM_DELAY = 0.05

AGENT_META = {
    "travel_agent": {"name": "Travel", "avatar": "🧳", "color": "#0ea5a6"},
    "news_agent":   {"name": "News",   "avatar": "📰", "color": "#ef4444"},
    "edumind_agent":{"name": "EduMind","avatar": "🎓", "color": "#f59e0b"},
    "med_agent":    {"name": "Medic",  "avatar": "🏥", "color": "#3b82f6"},
    "default":      {"name": "Agent",  "avatar": "🤖", "color": "#6b7280"},
}

# ---------- Helpers ----------
def escape_html(s: str) -> str:
    return html.escape(str(s))

def agent_html(agent_id: str, text: str) -> str:
    meta = AGENT_META.get(agent_id, AGENT_META["default"])
    avatar = meta["avatar"]
    color = meta["color"]
    name = meta["name"]
    safe_text = escape_html(text).replace("\n", "<br/>")
    return f"""
<div style="display:flex; gap:8px; margin:6px 0;">
  <div style="font-size:28px;">{avatar}</div>
  <div style="background:{color}22; border-left:4px solid {color}; border-radius:12px; padding:8px; max-width:75%; color:#fff;">
    <div style="font-size:12px; opacity:0.9; font-weight:600;">{name}</div>
    <div style="font-size:14px; line-height:1.35;">{safe_text}</div>
  </div>
</div>
"""

def user_html(text: str) -> str:
    safe_text = escape_html(text).replace("\n", "<br/>")
    return f'<div style="background:transparent; color:#fff; padding:8px; border-radius:0; max-width:75%; margin:6px 0;">{safe_text}</div>'


def chunk_string(s: str, n: int):
    if not s: yield ""
    words = s.split()
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > n and cur:
            yield " ".join(cur)
            cur = [w]
            cur_len = len(w)+1
        else:
            cur.append(w)
            cur_len += len(w)+1
    if cur:
        yield " ".join(cur)

def make_chat_message(role: str, content: str) -> Dict:
    return {"role": role, "content": content}

# ---------- Streaming handler ----------
def streaming_chat(user_message: str, history_state: List[Dict]):
    if history_state is None:
        history_state = []

    # Append user message
    history_state.append(make_chat_message("user", user_html(user_message)))
    yield history_state, {}

    # Generate plan
    try:
        plan = plan_task(user_message)
    except Exception:
        plan = [{"task": user_message, "agent": "edumind_agent"}]
    yield history_state, plan

    # Execute each step
    for step in plan:
        agent_id = step.get("agent", "edumind_agent")
        agent_name = AGENT_META.get(agent_id, AGENT_META["default"])["name"]
        accumulated = ""
        # Placeholder assistant message
        history_state.append(make_chat_message("assistant", f"{agent_name} is typing…"))
        index = len(history_state) - 1
        yield history_state, plan

        # Execute agent
        try:
            result_obj = execute_step(step)
            full_text = result_obj.get("result", str(result_obj))
        except Exception as e:
            full_text = f"Error executing agent {agent_id}: {e}"

        # Stream typing effect
        for chunk in chunk_string(full_text, CHUNK_SIZE):
            accumulated = f"{accumulated} {chunk}".strip()
            history_state[index]["content"] = agent_html(agent_id, accumulated)
            yield history_state, plan
            time.sleep(STREAM_DELAY)

    yield history_state, plan

# ---------- Dark mode via Markdown ----------
dark_mode_css = """
<style>
body {background:#0b1221; color:#e6eef8;}
.gradio-container {background:transparent;}
</style>
"""

# ---------- Build UI ----------
with gr.Blocks() as demo:
    gr.HTML(dark_mode_css)
    gr.Markdown("<h2 style='color:#cfe9ff'>🧠 AGENT VERSE - A Multi-Agent Chatbot </h2>")

    with gr.Row():
        with gr.Column(scale=3, min_width=600):
            chatbot = gr.Chatbot(label="Conversation")
            with gr.Row():
                txt = gr.Textbox(show_label=False, placeholder="Type your message and press Enter", lines=1)
                submit = gr.Button("Send")
            history_state = gr.State([])

        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 🔍 Planner Debug")
            debug = gr.JSON(value={}, label="Latest Plan / Steps")

    submit.click(
    streaming_chat,
    inputs=[txt, history_state],
    outputs=[chatbot, debug],
    show_progress=True
).then(lambda: "", None, txt)    # CLEAR TEXTBOX

    txt.submit(
    streaming_chat,
    inputs=[txt, history_state],
    outputs=[chatbot, debug],
    show_progress=True
).then(lambda: "", None, txt)    # CLEAR TEXTBOX


if __name__ == "__main__":
    demo.queue().launch()
