from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(model="gpt-3.5-turbo",api_key=os.getenv("OPENAI_API_KEY"))
