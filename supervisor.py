from langgraph.graph import StateGraph,END
from langfuse import observe
from langgraph.checkpoint.memory import MemorySaver
from state import AbilifyState
from retrieval import child_chunk_creation,vectorstore_creation,hybrid_search_rerank,attach_parent_context
import os
from agents.clinical_agent import clinical_agent
from agents.drug_interaction_agent import drug_interaction_agent
from agents.safety_agent import safety_agent    
from agents.evaluation_agent import evaluation_agent
from llm.llm import llm


def question_checking(state:AbilifyState)->dict:
    query=state["query"]
    prompt=f"""You are a helpful agent that checks if the {query} is a valid question realted to abilify that can be answered.
    If the question valid, return "valid", if its not valid return "invalid".
    Output will be a dictionary, exmaple: {{"next":"valid"}} or {{"next":"invalid"}}
    query:{query}"""
    response=llm.invoke(prompt)
    if "valid" in response.content.lower():
        return {"next":"valid"}
    else:        
        return {"next":"invalid"}
    
def agent_decision(state:AbilifyState)->dict:
    prompt=f""" You are a routing agent for an Abilify clinical Q&A system.

    Based on the user query decide which specialist agent should handle it.

    Available agents:
    - clinical_agent: handles general Abilify information, 
    side effects, dosage, how it works, clinical trials
    
    - drug_interaction_agent: handles questions about combining 
    Abilify with other medications, drug interactions, 
    contraindications with other drugs
    
    - safety_agent: handles warnings, overdose, black box warnings,
    special populations like pregnant women, elderly, children

    Query: {state["query"]}

    Return ONLY one of: clinical_agent, drug_interaction_agent, safety_agent
    Nothing else.
    return dictionary format, example: {{"next":"clinical_agent"}}"""



def evaluation_agent(state:AbilifyState)->dict:
    response=evaluation_agent.invoke(query=state["query"],answer=state["current_answer"],context=state["retrieved_context"])
    if "sufficient" in response.content.lower():
        return {"next":"Satisfied","eval_result":" Satisfied","retry_count":state["retry_count"]}
    else:
        return {"next":"Unsatisfied", "eval_result":"Unsatisfied","retry_count":state["retry_count"]+1}

#def clinical_agent(state:AbilifyState)->dict:


graph=StateGraph(AbilifyState)
graph.add_node("question_checking",question_checking)
graph.add_node("agent_decision",agent_decision)
graph.add_node("clinical_agent",clinical_agent)
graph.add_node("drug_interaction_agent",drug_interaction_agent)
graph.add_node("safety_agent",safety_agent)
graph.add_node("evaluation_agent",evaluation_agent)

graph.set_entry_point("question_checking")

graph.add_conditional_edges("question_checking",lambda state:state["next"],{"valid":"agent_decision","invalid":END})
graph.add_conditional_edges("agent_decision", lambda state:state["next"],{"clinical_agent":"clinical_agent","drug_interaction_agent":"drug_interaction_agent","safety_agent":"safety_agent"})
graph.add_edge("clinical_agent","evaluation_agent")
graph.add_edge("drug_interaction_agent","evaluation_agent")
graph.add_edge("safety_agent","evaluation_agent")
def route_after_evaluation(state: AbilifyState) -> str:
    if state["eval_result"] == "Satisfied":
        return "END"
    return state["previous_agent"]  # dynamic at runtime

graph.add_conditional_edges(
    "evaluation_agent",
    route_after_evaluation,  # reads state at runtime
    {
        "END": END,
        "clinical_agent": "clinical_agent",
        "drug_interaction_agent": "drug_interaction_agent",
        "safety_agent": "safety_agent"
    }
)

