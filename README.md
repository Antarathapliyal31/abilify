# Abilify Clinical Q&A Agent

An agentic RAG system for answering clinical questions about Abilify (aripiprazole) using LangGraph, LangChain, and MCP integration.

## Overview

This system uses a multi-agent architecture to answer clinical questions about Abilify (aripiprazole) from Otsuka Pharmaceutical. It combines FDA label document retrieval with PubMed medical literature search, orchestrated by LangGraph with an inline evaluation agent for quality assurance.

## Architecture

```
User Query
    ↓
Question Checking — validates Abilify relevance
    ↓
Agent Decision — routes to specialist
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
LLM:              OpenAI GPT-3.5-turbo
Embeddings:       OpenAI text-embedding-3-small
Vector Store:     ChromaDB (persistent)
Keyword Search:   BM25Retriever
Reranking:        Cohere rerank-english-v3.0
PDF Loading:      PyMuPDF
MCP Integration:  Public PubMed MCP server
Observability:    LangFuse
```

## Guardrails

1. XML tagging — prevents prompt injection
2. Answer only from context — prevents hallucination
3. Mandatory medical disclaimer — every response
4. Source citation required — reduces hallucination
5. No diagnosis — never recommends starting or stopping medication
6. Scope restriction — only answers Abilify related questions

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
│   ├── __init__.py
│   └── llm.py
├── mcp_client_pubmed/
│   ├── __init__.py
│   └── mcp_client.py
├── docs/
│   └── RAG_doc_med.pdf
├── retrieval.py
├── state.py
├── supervisor.py
├── main.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Antarathapliyal31/abilify.git
cd abilify
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```
OPENAI_API_KEY=your-openai-api-key
COHERE_API_KEY=your-cohere-api-key
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 5. Add Abilify FDA label PDF

Place the Abilify FDA label PDF in the `docs/` folder named `RAG_doc_med.pdf`.

### 6. Run

```bash
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
1. Question checking validates if query is Abilify related
2. Agent decision routes to appropriate specialist agent
3. Specialist agent rewrites query and retrieves from FDA label
4. If FDA context insufficient agent calls PubMed MCP server
5. Evaluation agent checks faithfulness, completeness, disclaimer, citation
6. If insufficient system retries up to 2 times
7. Final answer returned with medical disclaimer

## MCP Integration

The system connects to a public PubMed MCP server for real-time medical literature search:

```
Server: https://pubmed.caseyjhand.com/mcp
Tool: pubmed_search_articles
Usage: Fallback when FDA label context is insufficient
```

No authentication required for this public server.

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
- Chroma flat metadata requirement — fixed LLM prompt to return flat JSON only
- @observe decorator incompatibility with @tool — removed from tool functions
- JSON parsing with extra appended text — extracted JSON by bracket matching
- State field name mismatches — standardized all field names to match state.py
- BM25 persistence across restarts — saved child chunks to pickle file
- asyncio context propagation with LangFuse — applied @observe only to sync wrappers
- LLM returning JSON string instead of plain agent name — added defensive parsing

## Disclaimer

This system is for informational purposes only and does not constitute medical advice. Always consult a qualified medical professional before making any treatment decisions regarding Abilify or any other medication.
