from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
import json
import os

# Load environment variables
load_dotenv()
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Initialize LLM
llm = ChatLiteLLM(
    model="ollama/deepseek-r1:1.5b",
    streaming=True,
    temperature=0.7
)

# Tavily search tool
search_tool = TavilySearchResults(api_key=TAVILY_KEY, max_results=3)

# Optional pre-model hook
def pre_model_hook(state: dict) -> dict:
    print("🔧 Pre-model hook called")
    return state

# Structured output format
class ResponseFormat(BaseModel):
    result: str

# Build ReAct agent
agent = create_react_agent(
    model=llm,
    tools=[search_tool],
    pre_model_hook=pre_model_hook,
    response_format=ResponseFormat
)

def news_planner(prompt: str, max_chars: int = 300) -> str:
    """
    Execute the news agent and return concise, readable results with links.
    
    Args:
        prompt (str): User query for news.
        max_chars (int): Maximum characters to show per article.

    Returns:
        str: Formatted, truncated news results.
    """
    # Execute Tavily search
    tool_result = search_tool.invoke({"query": prompt})

    # Handle both list and dict responses
    if isinstance(tool_result, list):
        results = tool_result
    elif isinstance(tool_result, dict) and "results" in tool_result:
        results = tool_result["results"]
    else:
        results = []

    # Format results concisely
    formatted = ""
    for res in results:
        title = res.get("title", "No title")
        content = res.get("content", "")
        url = res.get("url", "")
        
        # Truncate content to max_chars
        snippet = (content[:max_chars] + "...") if len(content) > max_chars else content
        
        formatted += f"🔹 {title}\n{snippet}\nRead more: {url}\n\n"

    return formatted.strip() or "No results found."



# Test run
if __name__ == "__main__":
    prompt = "latest news about AI"
    news = news_planner(prompt)
    print("\n📰 Formatted News Results:\n")
    print(news)
