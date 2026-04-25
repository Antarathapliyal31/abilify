from langchain.agents import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,HumanMessagePromptTemplate,SystemMessagePromptTemplate
import os
from retrieval import hybrid_search_rerank,attach_parent_context
from dotenv import load_dotenv
from langfuse import observe
load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo",api_key=os.getenv("OPENAI_API_KEY"))