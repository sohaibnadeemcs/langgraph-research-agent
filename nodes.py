"""
nodes.py
--------
Each function here is one "node" in the LangGraph graph. A node takes the
current AgentState, does some work (calls an LLM, calls a search API, etc.),
and returns a dict of the fields it wants to update in the state.

Nodes in this agent:
    1. planner_node      -> breaks the user's query into sub-questions
    2. search_node        -> runs a web search (Tavily) for each sub-question
    3. synthesizer_node   -> combines all findings into a final written summary
"""

import os
import json
from langchain_groq import ChatGroq
from tavily import TavilyClient
from state import AgentState

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
# Using Groq for fast, free-tier inference. Swap the model name for any other
# model listed at https://console.groq.com/docs/models if you want.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# 1. Planner node
# ---------------------------------------------------------------------------
def planner_node(state: AgentState) -> dict:
    """
    Takes the user's raw query and breaks it into 2-3 focused sub-questions.
    Breaking a broad question into smaller ones produces much better search
    results than searching the raw query directly.
    """
    prompt = f"""You are a research planning assistant.

Break the following question into 2 to 3 focused, standalone sub-questions
that, if answered, would let someone fully answer the original question.

Original question: "{state['query']}"

Respond ONLY with a valid JSON array of strings, nothing else.
Example format: ["sub-question 1", "sub-question 2", "sub-question 3"]
"""

    response = llm.invoke(prompt)
    raw_text = response.content.strip()

    # Models sometimes wrap JSON in markdown fences despite instructions.
    # Strip those defensively so json.loads doesn't blow up.
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        sub_questions = json.loads(raw_text)
        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError("Planner did not return a valid list")
    except (json.JSONDecodeError, ValueError):
        # Fallback: if parsing fails, just use the original query as-is
        # so the graph can still continue instead of crashing.
        sub_questions = [state["query"]]

    return {"sub_questions": sub_questions, "retry_count": 0}


# ---------------------------------------------------------------------------
# 2. Search node
# ---------------------------------------------------------------------------
def search_node(state: AgentState) -> dict:
    """
    Runs a Tavily web search for each sub-question and collects the results.
    Sets search_failed=True if nothing useful came back, which the graph
    uses to decide whether to retry or fail gracefully.
    """
    results = []

    for sub_q in state["sub_questions"]:
        try:
            response = tavily_client.search(
                query=sub_q,
                search_depth="basic",
                max_results=3,
            )
            combined_content = "\n".join(
                r.get("content", "") for r in response.get("results", [])
            )
            sources = [r.get("url", "") for r in response.get("results", [])]

            results.append({
                "sub_question": sub_q,
                "content": combined_content,
                "sources": sources,
            })
        except Exception as e:
            # Record the failure but keep going for the other sub-questions
            results.append({
                "sub_question": sub_q,
                "content": "",
                "sources": [],
            })

    # Consider the search "failed" only if EVERY sub-question came back empty
    search_failed = all(r["content"].strip() == "" for r in results)

    return {
        "search_results": results,
        "search_failed": search_failed,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 3. Synthesizer node
# ---------------------------------------------------------------------------
def synthesizer_node(state: AgentState) -> dict:
    """
    Takes all the collected search results and asks the LLM to write a
    clean, structured final answer: intro, key findings, sources.
    """
    context_blocks = []
    all_sources = set()

    for r in state["search_results"]:
        context_blocks.append(f"Sub-question: {r['sub_question']}\nFindings: {r['content']}")
        all_sources.update(r["sources"])

    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant writing a final answer for the user.

Original question: "{state['query']}"

Here is the research gathered from the web:

{context_text}

Write a clear, well-organized answer with:
1. A short 2-3 sentence overview
2. Key findings as bullet points
3. A brief closing takeaway

Do not mention "sub-questions" or your internal process. Write it as a
single polished answer for the end user.
"""

    response = llm.invoke(prompt)
    summary = response.content.strip()

    if all_sources:
        sources_list = "\n".join(f"- {s}" for s in all_sources if s)
        summary += f"\n\n**Sources:**\n{sources_list}"

    return {"final_summary": summary}


# ---------------------------------------------------------------------------
# 4. Failure node (used when search fails after max retries)
# ---------------------------------------------------------------------------
def failure_node(state: AgentState) -> dict:
    """
    Reached only if search kept failing after MAX_RETRIES. Gives the user
    a graceful message instead of crashing or returning an empty answer.
    """
    return {
        "final_summary": (
            "I wasn't able to find reliable information for this query "
            "after multiple attempts. Try rephrasing your question or "
            "checking your Tavily API key/quota."
        ),
        "error": "search_failed_max_retries",
    }
