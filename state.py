from typing import TypedDict, Annotated,List
from langgraph.graph.message import add_messages
class AbilifyState(TypedDict):
    query:str
    messages:Annotated[list,add_messages]
    retrieval_context:str
    current_answer:str
    eval_result:str
    final_answer:str
    retry_count:int
    next:str
    previous_agent: str