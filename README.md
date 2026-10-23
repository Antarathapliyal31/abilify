# Abilify Clinical Q&A Agent

An agentic RAG system for answering clinical questions about Abilify (aripiprazole) using LangGraph, LangChain, and MCP integration.

## Overview

This system uses a multi-agent architecture to answer clinical questions about Abilify (aripiprazole) from Otsuka Pharmaceutical. It combines FDA label document retrieval with PubMed medical literature search, orchestrated by LangGraph with an inline evaluation agent for quality assurance.

## Architecture

```
User Query
    ↓
PII Redaction — Presidio strips PII/PHI locally before any LLM sees the query
    ↓
Question Checking — validates Abilify relevance (query wrapped in XML tags)
    ↓
Agent Decision — routes to specialist (query wrapped in XML tags)
    ↓
Clinical Agent / Drug Interaction Agent / Safety Agent
    ↓
Evaluation Agent — checks quality with 4 tools
    ↓
Retry if insufficient (max 2 retries)
    ↓
Final Answer
```

## Agents

**Clinical Agent**
Handles general Abilify information, side effects, dosage, mechanism of action, and clinical trial data.
Tools: FDA label RAG retrieval, PubMed MCP search

**Drug Interaction Agent**
Handles questions about combining Abilify with other medications, drug interactions, and contraindications.
Tools: FDA label RAG retrieval, PubMed MCP search

**Safety Agent**
Handles warnings, overdose information, black box warnings, and special populations including pediatric, elderly, and pregnant patients.
Tools: FDA label RAG retrieval, PubMed MCP search

**Evaluation Agent**
Inline quality gate that autonomously evaluates answers before returning to user.
Tools: check_faithfulness, check_completeness, check_medical_disclaimer, check_source_citation

## RAG Pipeline

```
FDA Label PDF
    ↓
Parent Child Chunking (parent=1000 tokens, child=300 tokens)
    ↓
LLM Metadata Extraction per chunk
    ↓
Hybrid Search (BM25 + Vector, weights 0.6/0.4)
    ↓
Cohere Reranking (top 3 chunks)
    ↓
Parent Context Retrieval
    ↓
LLM Generation with Guardrails
```

## Tech Stack

```
Orchestration:    LangGraph
Agents:           LangChain AgentExecutor (langchain_classic)
LLM:              Anthropic Claude Haiku 4.5 via AWS Bedrock
Metadata LLM:     OpenAI GPT-3.5-turbo (ingestion-time only, flat-JSON tuned)
Embeddings:       OpenAI text-embedding-3-small
Vector Store:     ChromaDB (persistent)
Keyword Search:   BM25Retriever
Reranking:        Cohere rerank-english-v3.0
PDF Loading:      LangChain S3FileLoader (bucket: abilify-docs)
PII Redaction:    Microsoft Presidio + spaCy en_core_web_sm (local, no external calls)
MCP Integration:  Public PubMed MCP server
Observability:    LangFuse
```

## Guardrails

1. PII/PHI redaction — Presidio strips personal/health identifiers locally before any LLM call
2. XML tagging — user query wrapped in tags and treated as data, preventing prompt injection
3. Answer only from context — prevents hallucination
4. Mandatory medical disclaimer — returned as a structured field in every response
5. Source citation required — reduces hallucination
6. No diagnosis — never recommends starting or stopping medication
7. Scope restriction — only answers Abilify related questions

## Project Structure

```
abilify/
├── agents/
│   ├── __init__.py
│   ├── clinical_agent.py
│   ├── drug_interaction_agent.py
│   ├── safety_agent.py
│   └── evaluation_agent.py
├── llm/
│   └── llm.py
├── mcp_client_pubmed/
│   ├── __init__.py
│   └── mcp_client.py
├── docs/
│   └── RAG_doc_med.pdf
├── tests/
│   ├── eval_questions.json
│   ├── run_eval.py
│   └── test_redaction.py
├── .github/workflows/ci.yml
├── graph.py              # LangGraph state machine (was supervisor.py)
├── retrieval.py          # ingestion + hybrid search + rerank
├── redaction.py          # Presidio PII redaction (step 1 of the pipeline)
├── state.py              # AbilifyState TypedDict
├── app.py                # Streamlit frontend
├── main.py               # entry point
├── requirements.txt
├── LICENSE
├── .gitignore
├── .env                  # not committed
└── README.md
```

## Setup

### 1. Clone the repository

```
git clone https://github.com/Antarathapliyal31/abilify.git
cd abilify
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for Presidio redaction
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```
OPENAI_API_KEY=your-openai-api-key
COHERE_API_KEY=your-cohere-api-key
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
# AWS creds via default chain (env or ~/.aws/credentials) for Bedrock + S3
```

### 5. Make the FDA label PDF available

The ingestion pipeline reads the PDF from S3: bucket `abilify-docs`, key `RAG_doc_med.pdf`. Ensure your AWS credentials have read access, or swap in a local loader in `retrieval.py:document_loading`.

### 6. Run

```
python main.py
```

On first run the system builds the vector index automatically. This takes 5-10 minutes due to LLM metadata extraction per chunk. Subsequent runs load from disk instantly.

## How It Works

### Ingestion (first run only)

1. Loads Abilify FDA label PDF using PyMuPDF
2. Splits into parent chunks (1000 tokens) and child chunks (300 tokens)
3. Extracts structured metadata per chunk using GPT-3.5-turbo
4. Embeds child chunks using OpenAI text-embedding-3-small
5. Stores embeddings in ChromaDB and parent texts in JSON

### Query Time

1. Presidio redacts PII/PHI from the query locally (names, emails, phone, SSN, labeled DOB) before any LLM sees it; a drug name allow-list keeps terms like "Abilify" and "Prozac" intact
2. Question checking validates if the redacted query is Abilify related (query wrapped in XML tags, `<ENTITY>` placeholders stripped before classification)
3. Agent decision routes to the appropriate specialist agent (query wrapped in XML tags)
4. Specialist agent rewrites the query and retrieves from the FDA label
5. If FDA context is insufficient the agent calls the PubMed MCP server
6. Evaluation agent checks faithfulness, completeness, disclaimer, citation
7. If insufficient the system retries up to 2 times
8. Final answer returned with medical disclaimer

## PII / PHI Redaction

Before any user query reaches an LLM, it passes through a redaction node built on Microsoft Presidio, which runs entirely locally with the `en_core_web_sm` spaCy model (no external calls). Presidio combines regex recognizers for structured identifiers (SSN, email, phone, credit card) with local spaCy NER for unstructured PII (names, locations), plus a custom `DATE_OF_BIRTH` recognizer that catches labeled dates (`DOB 1985-03-12`, `born on March 12, 1985`) without touching bare years used in research queries ("after 2016"). A psychiatric-drug allow-list (Abilify, Prozac, Zoloft, Lexapro, lithium, …) prevents the small spaCy model from falsely tagging drug names as `PERSON`. Because detection is local and deterministic — never an LLM call — no personal or health data is transmitted to the model, its prompts, or the observability traces. This is the design a HIPAA-regulated deployment requires; for clinical identifiers such as MRNs, custom Presidio recognizers can be added.

## MCP Integration

The system connects to a public PubMed MCP server for real-time medical literature search:

```
Server: https://pubmed.caseyjhand.com/mcp
Tool: pubmed_search_articles
Usage: Fallback when FDA label context is insufficient
```

No authentication required for this public server.

## Frontend

A Streamlit UI is included at `app.py`:

```bash
streamlit run app.py
```

The UI shows the raw question, the Presidio-redacted version actually fed to the LLM, the routed agent, and the final answer. Session-scoped `thread_id` means the LangGraph checkpointer sees multi-turn history.

## Evaluation

An offline eval runs the full pipeline against 25 curated questions spanning the four agent paths plus PII and out-of-scope cases. Scoring is deterministic: routing match, keyword recall, disclaimer presence, and (for PII rows) redaction verification.

```bash
python tests/run_eval.py              # all 25 (writes tests/eval_results.json)
python tests/run_eval.py --limit 5    # first 5 only
python tests/run_eval.py --id pii-01  # a single row
```

Latest run (2026-10-22, Bedrock Claude Haiku 4.5):

| Category         | n | Routing | Disclaimer | Redaction | Avg keyword recall |
|------------------|---|---------|------------|-----------|--------------------|
| clinical         | 7 | 100%    | 100%       | 100%      | 0.92               |
| drug_interaction | 6 | 100%    | 83%        | 100%      | 0.50               |
| safety           | 7 | 100%    | 100%       | 100%      | 0.65               |
| pii_redaction    | 2 | 100%    | 100%       | 100%      | 0.75               |
| out_of_scope     | 3 | 100%    | 100%       | 100%      | 1.00               |
| **Overall**      | **25** | **100%** | **96%** | **100%** | **0.74**           |

`pii-01` was re-tested in isolation after fixing scope-checker placeholder handling (see "Known gaps" below). The other 24 results are from the single full run.

Known gaps:
- `interaction-02`: agent hit `max_iterations=3` before producing a parseable JSON block. Raising the iteration cap or tightening the prompt would recover this.
- Low keyword recall on some drug-interaction/safety rows reflects test strictness (answers are correct but don't repeat every expected keyword), not wrong answers.

Scope-checker PII robustness (fixed): earlier runs misclassified `pii-01` (a side-effects question prefixed with name/email/phone/DOB) as out-of-scope because the redacted placeholders confused the scope LLM. `question_checking` now strips `<ENTITY>` placeholders before classification; the specialist still sees the full redacted query. The DOB custom recognizer also wipes `DOB 1985-03-12`-style constructs without touching bare years like "after 2016".

CI runs the redaction unit tests only (`tests/test_redaction.py`); the end-to-end eval needs Bedrock/OpenAI/Cohere/S3 credentials and runs locally.

## Observability

All agent runs are traced in LangFuse:

- Full trace per query
- Token usage and cost per agent call
- Evaluation scores per run
- Latency per step
- Tool calls made by each agent

## Agentic RAG Pattern

This system implements agentic RAG where:

- Agent decides which source to retrieve from (PDF vs PubMed)
- Agent reformulates query for better retrieval
- Evaluation agent decides if retrieval quality is sufficient
- System retries autonomously if answer is insufficient
- Max 2 retries prevent infinite loops

## Challenges Solved During Development

- LangChain 1.x breaking changes — migrated to langchain_classic for agent support
- Chroma flat metadata requirement — fixed LLM prompt to return flat JSON only, with filter_complex_metadata as a safety net
- @observe decorator incompatibility with @tool — removed from tool functions
- JSON parsing with extra appended text — extracted JSON by bracket matching (first `{` to last `}`)
- Medical disclaimer lost before evaluation — the agent originally appended the disclaimer as loose text after the JSON, so the bracket-matching parse stripped it and it never reached the evaluation agent, causing check_medical_disclaimer to fail intermittently. Fixed by prompting the agent to return the disclaimer as a key inside the JSON, carrying it through a dedicated disclaimer state field, and having the evaluation agent read it from state
- State field name mismatches — standardized all field names to match state.py
- BM25 persistence across restarts — saved child chunks to pickle file
- asyncio context propagation with LangFuse — applied @observe only to sync wrappers
- LLM returning JSON string instead of plain agent name — added defensive parsing

## Disclaimer

This system is for informational purposes only and does not constitute medical advice. Always consult a qualified medical professional before making any treatment decisions regarding Abilify or any other medication.
