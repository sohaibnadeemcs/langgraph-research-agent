"""
main.py
-------
Simple command-line interface for the research agent.

Usage:
    python main.py
    (then type your question when prompted)
"""

from dotenv import load_dotenv
load_dotenv()

from graph import build_graph

def run():
    agent = build_graph()

    print("=" * 60)
    print("  LangGraph Research Agent")
    print("  Type a question and get a researched, sourced answer.")
    print("  Type 'exit' to quit.")
    print("=" * 60)

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue

        initial_state = {
            "query": query,
            "sub_questions": [],
            "search_results": [],
            "final_summary": "",
            "search_failed": False,
            "retry_count": 0,
            "error": None,
        }

        print("\nResearching... (this calls the LLM + web search, may take a few seconds)\n")
        result = agent.invoke(initial_state)

        print("-" * 60)
        print(result["final_summary"])
        print("-" * 60)


if __name__ == "__main__":
    run()
