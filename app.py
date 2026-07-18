"""
app.py
------
Streamlit UI for the research agent. Run with:

    streamlit run app.py

Gives you a shareable demo/screenshot for your portfolio instead of just a
terminal window.

Theme: the dark aurora color scheme is set via .streamlit/config.toml (a real
Streamlit theme, not CSS overrides), so every built-in widget — text input,
expander, buttons — gets correct, readable text color automatically. The CSS
below only adds the animated gradient background and a few decorative
accents on top of that base theme.
"""

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph import build_graph

st.set_page_config(page_title="LangGraph Research Agent", page_icon="🔎", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Force readable white/light text everywhere, regardless of theme load order */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] a,
[data-testid="stCaptionContainer"], [data-testid="stExpander"] p,
[data-testid="stExpander"] li {
    color: #f2f0ff !important;
}

[data-testid="stMarkdownContainer"] a {
    color: #4fd1c5 !important;
}

.stTextInput input {
    color: #1a0b2e !important;
}

/* Animated aurora gradient background on the whole app */
.stApp {
    background: linear-gradient(-45deg, #1a0b2e, #2d1b4e, #0f2847, #3b1550, #1a0b2e);
    background-size: 400% 400%;
    animation: auroraShift 18s ease infinite;
}
@keyframes auroraShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Title styling — gradient text */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #ff6ec7, #7c5cff, #4fd1c5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem !important;
}

/* Primary button — colorful animated gradient */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #ff6ec7, #7c5cff, #4fd1c5) !important;
    background-size: 200% 200% !important;
    animation: buttonGlow 6s ease infinite;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    box-shadow: 0 6px 24px rgba(124,92,255,0.4) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(255,110,199,0.5) !important;
}
@keyframes buttonGlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Expander (sub-questions) — colorful border accent */
[data-testid="stExpander"] {
    border: 1px solid rgba(79,209,197,0.4) !important;
    border-radius: 14px !important;
}

/* Answer section container — glassy card with gradient heading */
.answer-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(124,92,255,0.25);
    border-radius: 18px;
    padding: 24px 30px 6px;
    margin-top: 18px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
}
.answer-card h3 {
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(90deg, #4fd1c5, #7c5cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🔎 LangGraph Research Agent")
st.caption("Ask a question. The agent plans sub-questions, searches the web, and synthesizes a sourced answer.")

@st.cache_resource
def get_agent():
    return build_graph()

agent = get_agent()

query = st.text_input("Your question", placeholder="e.g. What are the pros and cons of nuclear energy?")

if st.button("Research", type="primary") and query:
    initial_state = {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "final_summary": "",
        "search_failed": False,
        "retry_count": 0,
        "error": None,
    }

    with st.spinner("Planning sub-questions, searching, and synthesizing..."):
        result = agent.invoke(initial_state)

    if result.get("sub_questions"):
        with st.expander("See how the agent broke down your question"):
            for sq in result["sub_questions"]:
                st.markdown(f"- {sq}")

    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown("### Answer")
    st.markdown(result["final_summary"])
    st.markdown('</div>', unsafe_allow_html=True)
