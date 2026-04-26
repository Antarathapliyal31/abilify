from langgraph.graph import StateGraph, END
from langfuse import observe
from langgraph.checkpoint.memory import MemorySaver
from state import AbilifyState
from agents.clinical_agent import executor as clinical_agent
from agents.drug_interaction_agent import executor as drug_interaction_agent
from agents.safety_agent import executor as safety_agent
from agents.evaluation_agent import executor as evaluation_agent
from llm.llm import llm
import json
from retrieval import hybrid_search_rerank, attach_parent_context
import os

@observe()
def question_checking(state: AbilifyState) -> dict:
    query = state["query"]
    prompt = f"""You are a abilify agent that checks if the {query} is a question related to abilify.
    If the question is valid, return "valid", if its not valid return "invalid".
    Example:
    query: what is teh weather in nJ today?
    output:{{"next": "invalid"}}
    query: what are the side effects of abilify in children?
    output:{{"next": "valid"}}
    This is a mandatory step.
    Be strict in checking if the query is valid or not.
    Output will be a dictionary, example: {{"next":"valid"}} or {{"next":"invalid"}}
    query:{query}"""
    response = llm.invoke(prompt)
    response_text = response.content.lower()
    if "invalid" in response_text:
        return {"next": "invalid"}
    elif "valid" in response_text:
        return {"next": "valid"}
    else:
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
    state["retrieved_context"] = retrieved_context

    response = clinical_agent.invoke({
        "query": query,
        "retrieved_context": retrieved_context
    })
    output = response["output"]
    print(f"DEBUG output: {output[:100]}")  # add this
    print(f"DEBUG type: {type(output)}")    # add this
    
    try:
        clean_output = output.split("Consult a qualified")[0].strip()
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        print(f"DEBUG json_str: {json_str[:100]}")  # add this
        result = json.loads(json_str)
        print(f"DEBUG result: {result}")            # add this
        
        if result.get("found_info") == False:
            return {
                "next": "Unsatisfied",
                "current_answer": result["answer"],
                "previous_agent": "clinical_agent"
            }
        print(f"DEBUG returning: {result.get('found_info')}")

        return {
            "next": "Satisfied",
            "current_answer": result["answer"],
            "previous_agent": "clinical_agent"
        }
    except Exception as e:
        print(f"DEBUG exception: {e}")              # add this
        return {
            "next": "Satisfied",
            "current_answer": output,
            "previous_agent": "clinical_agent"
        }

@observe()
def drug_interaction__agent(state: AbilifyState) -> dict:
    query = state["query"]
    results = hybrid_search_rerank(query)
    retrieved_context = attach_parent_context(results)
    state["retrieved_context"] = retrieved_context
    response = drug_interaction_agent.invoke({
        "query": state["query"],
        "retrieved_context": state["retrieved_context"]
    })
    output = response["output"]
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        result = json.loads(json_str)
        if result.get("found_info") == False:
            return {
                "next": "Unsatisfied",
                "current_answer": result["answer"],
                "previous_agent": "drug_interaction_agent"
            }
        return {
            "next": "Satisfied",
            "current_answer": result["answer"],
            "previous_agent": "drug_interaction_agent"
        }
    except:
        return {
            "next": "Satisfied",
            "current_answer": output,
            "previous_agent": "drug_interaction_agent"
        }

@observe()
def safety__agent(state: AbilifyState) -> dict:
    query = state["query"]
    results = hybrid_search_rerank(query)
    retrieved_context = attach_parent_context(results)
    state["retrieved_context"] = retrieved_context
    response = safety_agent.invoke({
        "query": state["query"],
        "retrieved_context": state["retrieved_context"]
    })
    output = response["output"]
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        result = json.loads(json_str)
        if result.get("found_info") == False:
            return {
                "next": "Unsatisfied",
                "current_answer": result["answer"],
                "previous_agent": "safety_agent"
            }
        return {
            "next": "Satisfied",
            "current_answer": result["answer"],
            "previous_agent": "safety_agent"
        }
    except:
        return {
            "next": "Satisfied",
            "current_answer": output,
            "previous_agent": "safety_agent"
        }

@observe()
def evaluation__agent(state: AbilifyState) -> dict:
    print(f"CURRENT ANSWER: {state['current_answer'][:200]}")
    query = state["query"]

    response = evaluation_agent.invoke({
        "question": query,
        "answer": state["current_answer"],
        "context": state.get("retrieved_context", "")
    })
    output = response["output"]
    
    if "sufficient" in output.lower() and "insufficient" not in output.lower():
        return {
            "eval_result": "Satisfied",
            "final_answer": state["current_answer"],
            "retry_count": state.get("retry_count", 0)
        }
    else:
        retry_count = state.get("retry_count", 0) + 1
        result = {
            "eval_result": "Unsatisfied",
            "retry_count": retry_count
        }
        if retry_count >= 2:
            result["final_answer"] = state.get("current_answer",
                "I could not find sufficient information. Please consult your doctor.")
        return result
@observe()
def route_after_evaluation(state: AbilifyState) -> str:
    if state.get("eval_result") == "Satisfied":
        return "END"
    if state.get("retry_count", 0) >= 2:
        return "END"
    return state["previous_agent"]

def supervisor_agent(state: AbilifyState) -> str:
    state["final_answer"]= "Donot have enough information for this query. Please consult a medical professional or your doctor for more information."
    return "END"

# Build graph
graph = StateGraph(AbilifyState)

graph.add_node("question_checking", question_checking)
graph.add_node("agent_decision", agent_decision)
graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("clinical_agent", clinical__agent)
graph.add_node("drug_interaction_agent", drug_interaction__agent)
graph.add_node("safety_agent", safety__agent)
graph.add_node("evaluation_agent", evaluation__agent)


graph.set_entry_point("question_checking")

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
