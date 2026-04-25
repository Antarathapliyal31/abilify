from langchain.agents import create_openai_tools_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,HumanMessagePromptTemplate,SystemMessagePromptTemplate
import os
from langfuse import observe
from langchain_core.tools import tool
from llm.llm import llm

@tool
def check_faithfulness(question: str, answer: str, context: str) -> str:
    """Check if every claim in the answer is supported 
    by the retrieved context. Use this to detect hallucination."""
    prompt=f"""You are a clinical fact-checker evaluating medical information accuracy.

    Your task is to check whether every claim in the answer is directly supported by the provided context.

    Definitions:
    FAITHFUL: claim explicitly stated or clearly implied by context
    UNFAITHFUL: claim introduces information not in context or contradicts context

    Examples:
    Context: "Abilify 10mg recommended for adults"
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
    
@tool
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

@tool
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

@tool
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

prompt=ChatPromptTemplate.from_messages([("System","""You are a strict clinical answer quality evaluator for an Abilify medical Q&A system.

Your job is to evaluate whether a clinical answer is good enough to return to a patient.

You have four evaluation tools:
- check_faithfulness: checks if claims are grounded in context
- check_completeness: checks if question is fully answered
- check_medical_disclaimer: checks if safety disclaimer present
- check_source_citation: checks if claims are cited

IMPORTANT RULES:
1. Start with check_faithfulness — if score is below 0.5 
   the answer is immediately INSUFFICIENT, stop there
2. Only use remaining tools if faithfulness passes
3. An answer needs ALL criteria to pass to be SUFFICIENT
4. Be strict — this is medical information

Final output must be exactly:
VERDICT: SUFFICIENT or INSUFFICIENT""" ),
MessagesPlaceholder(variable_name="agent_scratchpad")])

tools=[check_faithfulness,check_completeness,check_medical_disclaimer,check_source_citation]
evaluation_agent=create_openai_tools_agent(llm=llm,tools=tools,prompt=prompt)
evaluator=AgentExecutor(agent=evaluation_agent,tools=tools,max_iterations=3,handle_parsing_errors=True,verbose=True)
