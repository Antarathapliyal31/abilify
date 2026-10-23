import json
import logging
import os
import re

from langgraph.graph import END, StateGraph
from langfuse import observe
from langgraph.checkpoint.memory import MemorySaver

from state import AbilifyState
from agents.clinical_agent import executor as clinical_agent
from agents.drug_interaction_agent import executor as drug_interaction_agent
from agents.safety_agent import executor as safety_agent
from agents.evaluation_agent import evaluate
from llm.llm import llm
from retrieval import attach_parent_context, hybrid_search_rerank
from redaction import redact

logger = logging.getLogger(__name__)


def _as_text(output) -> str:
    """Normalize AgentExecutor output across providers.

    - OpenAI function-calling returns a plain str
    - Anthropic/Bedrock tool_use returns a list of content blocks like
      [{"type": "text", "text": "..."}, ...]
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(output)


_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+>")


def _strip_placeholders(text: str) -> str:
    """Drop `<PERSON>`-style redaction tokens so the scope-checker LLM sees
    only the semantic question. Specialists still get the full redacted
    text (placeholders preserve structure for retrieval queries)."""
    cleaned = _PLACEHOLDER_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


@observe()
def redact_pii(state: AbilifyState) -> dict:
    """Redact PII from the user query before any LLM sees it.

    Keeps the raw input in `original_query` for display/audit; downstream
    nodes (scope checking, routing, specialists) see only the redacted text.
    """
    raw = state["query"]
    cleaned = redact(raw)
    return {"original_query": raw, "query": cleaned}

@observe()
def question_checking(state: AbilifyState) -> dict:
    cleaned = _strip_placeholders(state["query"])
    prompt = f"""You are a scope classifier for an Abilify (aripiprazole) clinical Q&A system.

Decide whether the user is asking a question about Abilify or aripiprazole —
its uses, side effects, dosing, drug interactions, warnings, pregnancy,
pediatric use, overdose, or related clinical topics. Personal details
(names, dates, patient info) are not relevant to this decision; focus on
the medical question itself.

Examples:
- "What are the side effects of Abilify?" -> valid
- "Can Abilify be taken with Prozac?" -> valid
- "Is Abilify safe during pregnancy?" -> valid
- "DOB 1985, is Abilify safe?" -> valid
- "What is the weather in NJ?" -> invalid
- "Write me a Python function to sort a list." -> invalid

Return exactly one word: valid or invalid.

<query>{cleaned}</query>"""
    response = llm.invoke(prompt)
    text = response.content.strip().lower() if isinstance(response.content, str) else str(response.content).lower()
    if "invalid" in text:
        return {"next": "invalid"}
    if "valid" in text:
        return {"next": "valid"}
    logger.warning("question_checking returned unparseable: %r; defaulting to invalid", text[:80])
    return {"next": "invalid"}

@observe()
def agent_decision(state: AbilifyState) -> dict:
    prompt = f"""You are a routing agent for an Abilify clinical Q&A system.

    Based on the user query decide which specialist agent should handle it.

    Available agents:
    - clinical_agent: handles general Abilify information, 
    side effects, dosage, how it works, clinical trials
    
    - drug_interaction_agent: handles questions about combining 
    Abilify with other medications, drug interactions
    
    - safety_agent: handles warnings, overdose, black box warnings,
    special populations like pregnant women, elderly, children

    Query: {state["query"]}

    Return ONLY one of these exact words with no punctuation:
    clinical_agent
    drug_interaction_agent
    safety_agent"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    if "clinical_agent" in content:
        return {"next": "clinical_agent"}
    elif "drug_interaction_agent" in content:
        return {"next": "drug_interaction_agent"}
    elif "safety_agent" in content:
        return {"next": "safety_agent"}
    else:
        return {"next": "clinical_agent"}

@observe()
def clinical__agent(state: AbilifyState) -> dict:
    query = state["query"]
    results = hybrid_search_rerank(query)
    retrieved_context = attach_parent_context(results)

    response = clinical_agent.invoke({
        "query": query,
        "retrieved_context": retrieved_context
    })
    output = _as_text(response["output"])

    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        result = json.loads(json_str)

        if result.get("found_info") == False:
            return {
                "next": "Unsatisfied",
                "current_answer": result["answer"],
                "previous_agent": "clinical_agent",
                "retrieved_context": retrieved_context,
            }
        return {
            "next": "Satisfied",
            "current_answer": result["answer"],
            "previous_agent": "clinical_agent",
            "retrieved_context": retrieved_context,
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("clinical_agent output not parseable (%s); using raw text", type(e).__name__)
        return {
            "next": "Satisfied",
            "current_answer": output,
            "previous_agent": "clinical_agent",
            "retrieved_context": retrieved_context,
        }

@observe()
def drug_interaction__agent(state: AbilifyState) -> dict:
    query = state["query"]
    results = hybrid_search_rerank(query)
    retrieved_context = attach_parent_context(results)
    response = drug_interaction_agent.invoke({
        "query": query,
        "retrieved_context": retrieved_context,
    })
    output = _as_text(response["output"])
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        result = json.loads(json_str)
        if result.get("found_info") == False:
            return {
                "next": "Unsatisfied",
                "current_answer": result["answer"],
                "previous_agent": "drug_interaction_agent",
                "retrieved_context": retrieved_context,
            }
        return {
            "next": "Satisfied",
            "current_answer": result["answer"],
            "previous_agent": "drug_interaction_agent",
            "retrieved_context": retrieved_context,
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("drug_interaction_agent output not parseable (%s); using raw text", type(e).__name__)
        return {
            "next": "Satisfied",
            "current_answer": output,
            "previous_agent": "drug_interaction_agent",
            "retrieved_context": retrieved_context,
        }

@observe()
def safety__agent(state: AbilifyState) -> dict:
    query = state["query"]
    results = hybrid_search_rerank(query)
    retrieved_context = attach_parent_context(results)
    response = safety_agent.invoke({
        "query": query,
        "retrieved_context": retrieved_context,
    })
    output = _as_text(response["output"])
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        result = json.loads(json_str)
        if result.get("found_info") == False:
            return {
                "next": "Unsatisfied",
                "current_answer": result["answer"],
                "previous_agent": "safety_agent",
                "retrieved_context": retrieved_context,
            }
        return {
            "next": "Satisfied",
            "current_answer": result["answer"],
            "previous_agent": "safety_agent",
            "retrieved_context": retrieved_context,
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("safety_agent output not parseable (%s); using raw text", type(e).__name__)
        return {
            "next": "Satisfied",
            "current_answer": output,
            "previous_agent": "safety_agent",
            "retrieved_context": retrieved_context,
        }

@observe()
def evaluation__agent(state: AbilifyState) -> dict:
    verdict = evaluate(
        question=state["query"],
        answer=state["current_answer"],
        context=state.get("retrieved_context", ""),
    )
    if verdict["verdict"] == "SUFFICIENT":
        return {
            "eval_result": "Satisfied",
            "final_answer": state["current_answer"],
            "retry_count": state.get("retry_count", 0),
        }
    retry_count = state.get("retry_count", 0) + 1
    result = {"eval_result": "Unsatisfied", "retry_count": retry_count}
    if retry_count >= 2:
        result["final_answer"] = state.get(
            "current_answer",
            "I could not find sufficient information. Please consult your doctor.",
        )
    return result
@observe()
def route_after_evaluation(state: AbilifyState) -> str:
    if state.get("eval_result") == "Satisfied":
        return "END"
    if state.get("retry_count", 0) >= 2:
        return "END"
    return state["previous_agent"]

def supervisor_agent(state: AbilifyState) -> dict:
    return {
        "final_answer": "I don't have enough information for this query. Please consult a medical professional or your doctor for more information."
    }

# Build graph
graph = StateGraph(AbilifyState)

graph.add_node("redact_pii", redact_pii)
graph.add_node("question_checking", question_checking)
graph.add_node("agent_decision", agent_decision)
graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("clinical_agent", clinical__agent)
graph.add_node("drug_interaction_agent", drug_interaction__agent)
graph.add_node("safety_agent", safety__agent)
graph.add_node("evaluation_agent", evaluation__agent)


graph.set_entry_point("redact_pii")

graph.add_edge("redact_pii", "question_checking")

graph.add_conditional_edges(
    "question_checking",
    lambda state: state["next"],
    {"valid": "agent_decision", "invalid": "supervisor_agent"}
)

graph.add_conditional_edges(
    "agent_decision",
    lambda state: state["next"],
    {
        "clinical_agent": "clinical_agent",
        "drug_interaction_agent": "drug_interaction_agent",
        "safety_agent": "safety_agent"
    }
)

graph.add_conditional_edges(
    "clinical_agent",
    lambda state: state["next"],
    {"Satisfied": "evaluation_agent", "Unsatisfied": "supervisor_agent"}
)

graph.add_conditional_edges(
    "drug_interaction_agent",
    lambda state: state["next"],
    {"Satisfied": "evaluation_agent", "Unsatisfied": "supervisor_agent"}
)

graph.add_conditional_edges(
    "safety_agent",
    lambda state: state["next"],
    {"Satisfied": "evaluation_agent", "Unsatisfied": "supervisor_agent"}
)

graph.add_edge("supervisor_agent", END)

graph.add_conditional_edges(
    "evaluation_agent",
    route_after_evaluation,
    {
        "END": END,
        "clinical_agent": "clinical_agent",
        "drug_interaction_agent": "drug_interaction_agent",
        "safety_agent": "safety_agent"
    }
)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
