# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Agentic RAG system that answers clinical questions about Abilify (aripiprazole). The pipeline is a LangGraph state machine over specialist LangChain agents, backed by hybrid retrieval (BM25 + Chroma + Cohere rerank) over the FDA label PDF, with PubMed MCP as a fallback source.

## Running

```bash
source venv/bin/activate
python main.py          # entry point; query is hardcoded in main.py line 12
python retrieval.py     # exercise retrieval only (hardcoded query at bottom of file)
```

There is no `requirements.txt` in the repo despite what the README says — the `venv/` is the source of truth for installed deps. If adding a dep, install into the existing venv.

Required env vars in `.env` (loaded via `python-dotenv`):
- `OPENAI_API_KEY` — used by `retrieval.py` for embeddings (`text-embedding-3-small`) and metadata extraction (`gpt-3.5-turbo`)
- `COHERE_API_KEY` — reranker (`rerank-english-v3.0`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — tracing via `@observe()`
- AWS creds (default chain) — the central LLM at `llm/llm.py` is `ChatBedrock` on `us.anthropic.claude-haiku-4-5-20251001-v1:0` in `us-east-1`
- AWS creds also need read on S3 bucket `abilify-docs` key `RAG_doc_med.pdf` for first-time index build

## Architecture

### Two LLMs, on purpose

- `llm/llm.py` exports a single `llm` (Bedrock Claude Haiku 4.5) used by **every agent** and by the routing / question-checking / evaluation nodes in `graph.py`.
- `retrieval.py` instantiates its **own** `ChatOpenAI(gpt-3.5-turbo)` for per-chunk metadata extraction during ingestion. Do not route this through `llm/llm.py` casually — the metadata prompt is tuned to return flat JSON for Chroma (nested metadata will break `vectorstore.add_documents`).

### Graph (`graph.py`)

Entry: `question_checking` → `agent_decision` → one of `{clinical_agent, drug_interaction_agent, safety_agent}` → `evaluation_agent` → loop back to the same specialist on `Unsatisfied` (max 2 retries via `retry_count`), else END.

- `supervisor_agent` is **not** a router — it's the terminal fallback for invalid queries and for specialists that returned `found_info: false`. It writes a generic "consult a professional" message to `final_answer` and ends.
- Specialist nodes (`clinical__agent`, `drug_interaction__agent`, `safety__agent`) each: call `hybrid_search_rerank` → `attach_parent_context`, mutate `state["retrieved_context"]` directly, invoke the agent executor, then parse the first `{...}` block out of `response["output"]` and route on `found_info`. Parsing is best-effort; on any exception the raw output is treated as a successful answer.
- `evaluation__agent` decides SUFFICIENT/INSUFFICIENT by scanning for `"sufficient"` in its output while ensuring `"insufficient"` is absent. When at the retry cap, it populates `final_answer` with the last `current_answer` before ending.
- State is checkpointed in a `MemorySaver`; `main.py` generates a fresh `thread_id` per run.

### Retrieval (`retrieval.py`) — global state

`vectorstore` and `all_child_chunks` are **module-level globals** mutated by `initialize_retrieval()`. Nothing works until `initialize_retrieval()` has run in the current process; `main.py` calls it before `app.invoke`. If adding new entry points (scripts, tests, API handlers), call it there too.

Initialization has three branches by precedence:
1. All three of `./chroma_db/`, `parent_chunk.json`, `child_chunks.pkl` present → load from disk (instant).
2. Only `child_chunks.pkl` + `parent_chunk.json` present → rebuild Chroma from the pickle.
3. Otherwise → load PDF from S3, split (parent 1000/100, child 300/50), run LLM metadata per child chunk, persist. This is the slow path (5–10 minutes, costs real OpenAI tokens).

To force a rebuild, delete `chroma_db/`, `parent_chunk.json`, and `child_chunks.pkl` (all three are gitignored).

Hybrid search weighting is BM25 0.4 / vector 0.6, k=10 each, Cohere reranks to top 3. `attach_parent_context` then swaps each child chunk's content for its parent chunk (looked up via `metadata["parent_id"]` against `parent_chunk.json`) and joins with `\n\n`. **`attach_parent_context` reads `parent_chunk.json` from disk on every call** — if you change the parent store, flush writes before retrieving.

### Agents (`agents/*.py`)

All three specialists (`clinical_agent`, `drug_interaction_agent`, `safety_agent`) share the same shape:
- `create_openai_tools_agent(llm, tools, prompt)` + `AgentExecutor(max_iterations=3, handle_parsing_errors=True)`
- Tools: `pubmed_search` and `pubmed_fulltext` wrapping `mcp_client_pubmed/mcp_client.py`
- Prompt variables: `{query}`, `{retrieved_context}` — FDA retrieval is already done in the graph node, the agent does **not** retrieve FDA data itself; PubMed is strictly a fallback (though the clinical agent prompt forces PubMed when the query mentions "latest", "recent", "2023", "2024", etc.)
- Required output shape: a single JSON `{"answer": "...", "found_info": true|false}` with the medical disclaimer sentence as the final sentence **inside** `answer`, and every claim tagged `[Source: FDA Label]` or `[Source: PubMed PMID]`.

The evaluation agent (`agents/evaluation_agent.py`) is a separate agent executor with four `@tool` checkers (`check_faithfulness`, `check_completeness`, `check_medical_disclaimer`, `check_source_citation`). Each tool internally calls the Bedrock `llm` with a verdict prompt. The outer agent prompt directs it to short-circuit if faithfulness is below 0.5, and to always return `VERDICT: SUFFICIENT` or `VERDICT: INSUFFICIENT`.

### PubMed MCP (`mcp_client_pubmed/mcp_client.py`)

Public server `https://pubmed.caseyjhand.com/mcp`, no auth. Each call opens a fresh `streamable_http_client` + `ClientSession` and tears it down. The sync wrappers (`pubmed_mcp_client_search`, `pubmed_mcp_client_fulltext`) use `asyncio.run()` and are the entry points LangChain tools wrap — do **not** mark them `@observe()` on an inner async layer; LangFuse context does not propagate through `asyncio.run` cleanly (see README "Challenges Solved").

## Conventions worth preserving

- `@observe()` goes on **sync** functions only (not on `@tool`-decorated functions, not on inner async coroutines). This is a real constraint, not stylistic.
- State field names in `state.py` are the contract — specialist nodes return partial dicts with `next`, `current_answer`, `previous_agent`; the evaluator returns `eval_result`, `final_answer`, `retry_count`. The field `retrieval_context` in `state.py` is spelled differently from the `retrieved_context` written by specialist nodes — both exist; don't "fix" one without the other.
- Chroma requires flat metadata. The metadata extraction prompt enforces this and the loader strips non-primitive values; keep it that way.
- The README mentions `supervisor.py` — it was renamed to `graph.py` (commit `67644ad0`). `supervisor_agent` as a *node name* inside the graph still refers to the fallback terminal node.
