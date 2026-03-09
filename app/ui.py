"""
ui.py: Streamlit UI for StudioRAG.
Provides a chat interface with source attribution and retrieval latency for animation/VFX queries.
"""

import streamlit as st
from rag import get_vectorstore, get_retriever, get_llm, build_rag_chain, retrieve_sources

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudioRAG",
    page_icon="🎬",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #7c6af7;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .source-label {
        color: #7c6af7;
        font-weight: bold;
    }
    .snippet {
        color: #cdd6f4;
        font-style: italic;
    }
    .latency-badge {
        display: inline-block;
        background: #1e1e2e;
        border: 1px solid #7c6af7;
        color: #7c6af7;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎬 StudioRAG")
st.caption("Semantic Q&A over animation & VFX production documents · Powered by Claude 3.5 Haiku + ChromaDB")
st.divider()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    with st.spinner("Loading vector store and models..."):
        try:
            vs = get_vectorstore()
            retriever = get_retriever(vs)
            llm = get_llm()
            chain = build_rag_chain(retriever, llm)
            st.session_state.vs = vs
            st.session_state.chain = chain
            st.session_state.rag_ready = True
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {e}")
            st.info("Make sure ANTHROPIC_API_KEY is set and you have run `python ingest.py`.")
            st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    show_sources = st.toggle("Show source chunks", value=True)
    st.divider()

    st.header("💡 Example queries")
    example_queries = [
        "What are the stages of an animation pipeline?",
        "How does RenderMan XPU handle CPU and GPU rendering?",
        "What is motion capture retargeting?",
        "How are VFX image sequences named across studios?",
        "What is USD and how is it used in production pipelines?",
        "How does denoising work for deep compositing?",
        "What tools are used for procedural look development?",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
col_chat, col_sources = st.columns([2, 1]) if show_sources else (st.container(), None)

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about the animation/VFX production pipeline...")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(query)

    with col_chat:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                try:
                    sources, latency_ms = retrieve_sources(st.session_state.vs, query)
                    answer = st.session_state.chain.invoke(query)

                    st.markdown(
                        f'<span class="latency-badge">⚡ Retrieved in {latency_ms:.1f} ms</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    if show_sources and col_sources:
                        with col_sources:
                            st.subheader("📄 Source chunks")
                            for s in sources:
                                st.markdown(f"""
<div class="source-card">
<span class="source-label">📎 {s['source']} · p.{s['page']}</span><br>
<span class="snippet">{s['snippet']}…</span>
</div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.info("Check that your ANTHROPIC_API_KEY is valid.")