# 🔎 LangGraph Research Agent

A multi-step agentic AI workflow that plans, searches the web, and synthesizes a sourced research answer — built with **LangGraph** to demonstrate stateful, conditional agent orchestration rather than a single-shot LLM call.

**[🚀 Live Demo](https://langgraph-research-agent-live.streamlit.app/)** &nbsp;·&nbsp; Run locally: `streamlit run app.py`

---

## How it works

Instead of sending a user's question straight to an LLM, this agent breaks the task into a graph of specialized steps — each one able to react to what happened before it:

```
   ┌───────────┐      ┌────────┐      ┌──────────────┐
   │  Planner  │ ───▶ │ Search │ ───▶ │  Synthesizer │ ───▶ END
   └───────────┘      └────────┘      └──────────────┘
                          │  ▲
                 (search failed,     
                  retries left)
                          └──┘ retry
                          │
                 (search failed,
                  no retries left)
                          ▼
                     ┌─────────┐
                     │ Failure │ ───▶ END
                     └─────────┘
```

1. **Planner** — breaks the user's question into 2–3 focused sub-questions using Groq (`llama-3.3-70b-versatile`). Narrower sub-questions return far better search results than the raw query.
2. **Search** — runs each sub-question through the Tavily search API and collects results.
3. **Conditional routing** — if search comes back empty, the graph retries (up to 2 times) before giving up gracefully, instead of crashing or returning a hollow answer.
4. **Synthesizer** — combines everything gathered into a single structured answer: overview, key findings, and sources.

This conditional retry logic is the actual point of using LangGraph here — a plain script can't loop back and react to a bad search result the way a graph with edges and routing functions can.

## Tech Stack

- **LangGraph** / **LangChain** — graph orchestration
- **Groq API** (`llama-3.3-70b-versatile`) — fast LLM inference for planning and synthesis
- **Tavily API** — real-time web search
- **Streamlit** — web UI
- **python-dotenv** — local environment config

## Run it locally

```bash
git clone https://github.com/sohaibnadeemcs/langgraph-research-agent.git
cd langgraph-research-agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then add your GROQ_API_KEY and TAVILY_API_KEY to .env
```

**Web UI:**
```bash
streamlit run app.py
```

**Command line:**
```bash
python main.py
```

## Project structure

```
├── app.py           # Streamlit UI
├── main.py          # CLI entry point
├── graph.py         # Wires nodes into a LangGraph StateGraph
├── nodes.py         # Planner, Search, Synthesizer, Failure node logic
├── state.py         # Shared AgentState schema
├── .streamlit/
│   └── config.toml  # Custom theme
└── requirements.txt
```

## Get free API keys

- Groq: [console.groq.com/keys](https://console.groq.com/keys)
- Tavily: [tavily.com](https://tavily.com)

---

Built by [Sohaib Nadeem](https://github.com/sohaibnadeemcs) — [LinkedIn](https://linkedin.com/in/sohaib-nadeem-pk)
