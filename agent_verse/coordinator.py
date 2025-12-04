# coordinator.py
from travel_agent import travel_planner
from news_agent import news_planner
from edumind import edumind_agent
from med_agent import med_planner, med_agent
from plan import plan_task

def execute_step(step: dict) -> dict:
    """
    Execute a single step from the planner output.
    """
    agent = step["agent"]

    if agent == "travel_agent":
        result = travel_planner(step["prompt"])
    elif agent == "news_agent":
        result = news_planner(step["prompt"])
    elif agent == "edumind_agent":
        # edumind_agent is a StructuredTool, call via .invoke()
        result = edumind_agent.invoke({"query": step["prompt"]})
        if hasattr(result, "content"):
            result = result.content
    elif agent == "med_agent":
        # med_agent is a StructuredTool (or wrapper), call via .invoke() when available
        try:
            result = med_agent.invoke({"query": step["prompt"]})
            if hasattr(result, "content"):
                result = result.content
        except AttributeError:
            # fallback to med_planner wrapper which returns a string
            result = med_planner(step["prompt"]) 
    else:
        result = f"No such agent: {agent}"

    return {"result": result}

def main():
    print("🧠 Local Coordinator ready!\n")
    while True:
        user_prompt = input("💬 You: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Step 1: Generate plan
        plan = plan_task(user_prompt)
        print("\n📋 Planner Output:", plan)

        # Step 2: Execute steps
        for step in plan:
            result = execute_step(step)
            print(f"\n✅ Result from {step['agent']}:\n{result['result']}\n")

if __name__ == "__main__":
    main()
