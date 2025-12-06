
from langchain_community.chat_models import ChatLiteLLM
import json

# Initialize LLM for task planning
planner_llm = ChatLiteLLM(
    model="ollama/deepseek-r1:8b",
    streaming=False,
    temperature=0
)

# ---------------- AGENT CARDS ----------------
AGENT_CARDS = {
    "travel_agent": {
        "description": "Provides travel guides, itineraries, local attractions, hotels, tourist statistics, and travel advice in Pakistan."
    },
    "news_agent": {
        "description": "Provides latest news, updates, summaries, and analysis from reliable sources."
    },
    "med_agent": {
        "description": "Answers medical and health-related queries, provides evidence-backed guidance from authoritative sources."
    },
    "edumind_agent": {
        "description": "Answers educational, scientific, and learning queries. Explains concepts, solves problems, tutors on topics, and provides knowledge from curated datasets."
    }
}

# ---------------- PLAN TASK ----------------
def plan_task(user_prompt: str) -> list:
    """
    Multi-task planner using agent descriptions only:
    - Splits multi-intent queries
    - Classifies each sub-task to the appropriate agent
    - Returns a list of execution steps
    """
    agent_info = "\n".join([f"{k}: {v['description']}" for k, v in AGENT_CARDS.items()])

    prompt = f"""
You are a multi-agent task planner.

Available agents:
{agent_info}

Instructions:
1. Split the user's message into separate tasks if it contains multiple requests.
2. Assign each task to the most suitable agent: travel_agent, news_agent, or edumind_agent.
3. Return ONLY a JSON list in this format:

[
  {{"task": "...", "agent": "travel_agent"}},
  {{"task": "...", "agent": "edumind_agent"}}
]

User message: "{user_prompt}"
"""

    # Call the LLM
    response = planner_llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    # Parse JSON safely
    try:
        tasks = json.loads(text)
        if not isinstance(tasks, list):
            raise ValueError("LLM output is not a list")
    except Exception:
        # Fallback: treat entire prompt as a learning query
        tasks = [{"task": user_prompt, "agent": "edumind_agent"}]

    # Convert to execution plan steps
    plan = []
    for t in tasks:
        agent = t.get("agent", "edumind_agent").lower().strip()
        task_text = t.get("task", user_prompt)
        plan.append({
            "agent": agent,
            "type": "travel_plan" if agent == "travel_agent" else "news_brief" if agent == "news_agent" else "learning_query",
            "prompt": task_text,
            "arguments": {},
            "context_from_step": None
        })

    return plan
