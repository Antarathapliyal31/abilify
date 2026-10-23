"""Streamlit frontend for the Abilify Q&A system.

Run with:
    streamlit run app.py

Shows the original question, the redacted version Presidio produced, the
routed agent, and the final answer. Keeps the LangGraph thread_id across
turns within a single Streamlit session so the checkpointer can see history.
"""
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Abilify Clinical Q&A", page_icon=None, layout="centered")
st.title("Abilify Clinical Q&A")
st.caption(
    "Agentic RAG over the Abilify FDA label with a PubMed fallback. "
    "Patient PII is redacted before any LLM sees the query."
)


@st.cache_resource(show_spinner="Loading retrieval index...")
def _bootstrap():
    """One-time startup: build/load the vectorstore and compile the graph."""
    from retrieval import initialize_retrieval
    from graph import app
    from redaction import redact

    initialize_retrieval()
    return app, redact


app, redact = _bootstrap()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

question = st.text_area(
    "Ask a question about Abilify (aripiprazole):",
    height=120,
    placeholder="e.g. What are the common side effects of Abilify in adults?",
)

col1, col2 = st.columns([1, 1])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col2:
    if st.button("New thread", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.caption(f"thread_id: `{st.session_state.thread_id}`")

if submit and question.strip():
    with st.spinner("Thinking..."):
        preview_redacted = redact(question)
        if preview_redacted != question:
            with st.expander("Redacted query (what the LLM sees)", expanded=True):
                st.code(preview_redacted)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        try:
            result = app.invoke({"query": question}, config=config)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

    st.subheader("Answer")
    final = result.get("final_answer") or result.get("current_answer") or ""
    st.markdown(final)

    with st.expander("Debug"):
        st.json({
            "previous_agent": result.get("previous_agent"),
            "eval_result": result.get("eval_result"),
            "retry_count": result.get("retry_count"),
            "redacted_query": result.get("query"),
            "original_query": result.get("original_query"),
        })
