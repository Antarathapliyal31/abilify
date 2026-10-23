from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,HumanMessagePromptTemplate,SystemMessagePromptTemplate
import os
from langfuse import observe
from langchain_core.tools import tool
from llm.llm import llm

def check_faithfulness(question: str, answer: str, context: str = "") -> str:
    """Check if every claim in the answer is supported by the retrieved context."""
    if not context or not context.strip():
        return "VERDICT: UNFAITHFUL\nSCORE: 0.3\nREASON: no context provided\nUNFAITHFUL_CLAIMS: all"
    prompt=f"""You are a clinical fact-checker evaluating medical information accuracy.

    Your task is to check whether every claim in the answer is directly supported by the provided context.

    Definitions:
    FAITHFUL: claim explicitly stated or clearly implied by context
    UNFAITHFUL: claim introduces information not in context or contradicts context

    Examples:
    Context: "Abilify 10mg srecommended for adults"
    Answer: "Abilify 15mg recommended for adults"  
    Verdict: UNFAITHFUL — wrong dosage

    Context: "Abilify 10mg recommended for adults"
    Answer: "Adults take Abilify at 10mg"
    Verdict: FAITHFUL — accurately reflects context

    Now evaluate:
    Question: {question}
    Answer: {answer}
    Context: {context}

    Return exactly:
    VERDICT: FAITHFUL or UNFAITHFUL
    SCORE: 0.0 to 1.0
    REASON: one sentence
    UNFAITHFUL_CLAIMS: list or NONE """
    response=llm.invoke(prompt)
    return response.content

def check_completeness(question: str, answer: str) -> str:
    """Check if the answer fully addresses all parts of the question. Use this to detect incomplete answers."""
    prompt=f"""You are a clinical completeness evalutaor assesing clinical information completeness.
    Your task is to check if the asnwer fully adresses all parts of the question.
    Definitions:
    COMPLETE: answer addresses every aspect of the question with sufficient detail
    INCOMPLETE: answer misses key components of the question or lacks necessary detail
    Examples:
    Question: "What are the side effects of Abilify?"
    Answer: "Abilify can cause weight gain and drowsiness."
    Verdict: INCOMPLETE — misses other common side effects like nausea, dizziness       
    Question: "What are the side effects of Abilify?"
    Answer: "Abilify can cause weight gain, drowsiness, nausea, and dizziness."
    Verdict: COMPLETE — addresses multiple common side effects      
    Now evaluate:
    Question: {question}
    Answer: {answer}    
    Return exactly:
    VERDICT: COMPLETE or INCOMPLETE
    SCORE: 0.0 to 1.0
    REASON: one sentence
    MISSING_INFO: list or NONE"""
    response=llm.invoke(prompt)
    return response.content


def check_medical_disclaimer(answer: str) -> str:
    """Check if the answer contains appropriate medical disclaimer advising to consult a doctor."""
    prompt=f"""You are a clinical disclaimer evaluator.
    Your task is to check if the answer contains an appropriate medical disclaimer advising to consult a doctor.
    Definitions:
    PRESENT: answer includes a medical disclaimer
    MISSING: answer does not include a medical disclaimer
    Examples:
    Answer: "Abilify can cause weight gain and drowsiness." [Disclaimer: Consult your doctor before taking this medication or any treatment decisions.]
    Verdict: PRESENT — includes medical disclaimer
    Answer: "Abilify can cause weight gain and drowsiness."
    Verdict: MISSING — no medical disclaimer
    Now evaluate:
    Answer: {answer}
    Return exactly:
    VERDICT: PRESENT or MISSING
    SCORE: 0.0 to 1.0
    REASON: one sentence"""
    response=llm.invoke(prompt)
    return response.content


def check_source_citation(answer: str) -> str:
    """Check if the answer cites sources for clinical claims."""
    prompt=f"""You are a clinical source citation evaluator.
    Your task is to check if the answer cites sources for clinical claims.
    Definitions:
    CITED: answer includes source citations for clinical claims
    NOT_CITED: answer does not include source citations for clinical claims
    Examples:
    Answer: "Abilify can cause weight gain and drowsiness." [Source: FDA]
    Verdict: CITED — includes source citation
    Answer: "Abilify can cause weight gain and drowsiness."
    Verdict: NOT_CITED — no source citation
    Now evaluate:
    Answer: {answer}
    Return exactly:
    VERDICT: CITED or NOT_CITED
    SCORE: 0.0 to 1.0
    REASON: one sentence"""
    response=llm.invoke(prompt)
    return response.content

def _score_from(text: str, default: float = 0.0) -> float:
    """Extract the first `SCORE: X.Y` from a check-tool response."""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                return default
    return default


def evaluate(question: str, answer: str, context: str) -> dict:
    """Deterministic evaluator — runs all four checks and returns a verdict.

    Returns {"verdict": "SUFFICIENT"|"INSUFFICIENT", "faithfulness": ..., ...}.
    Short-circuits on faithfulness < 0.5 to save tokens on obviously bad
    answers. Each check calls Bedrock once; expect ~2-4 LLM calls per eval.
    """
    f = check_faithfulness(question, answer, context)
    f_score = _score_from(f)
    if f_score < 0.5:
        return {
            "verdict": "INSUFFICIENT",
            "faithfulness": f,
            "completeness": None,
            "disclaimer": None,
            "citation": None,
            "reason": "faithfulness_below_0.5",
        }

    c = check_completeness(question, answer)
    d = check_medical_disclaimer(answer)
    s = check_source_citation(answer)

    passes = (
        f_score >= 0.5
        and _score_from(c) >= 0.5
        and "PRESENT" in d.upper()
        and "CITED" in s.upper() and "NOT_CITED" not in s.upper()
    )
    return {
        "verdict": "SUFFICIENT" if passes else "INSUFFICIENT",
        "faithfulness": f,
        "completeness": c,
        "disclaimer": d,
        "citation": s,
    }
