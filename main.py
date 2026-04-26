from retrieval import initialize_retrieval
from supervisor import app
from dotenv import load_dotenv
from langfuse import observe
load_dotenv()
import uuid
@observe()
def main():
    initialize_retrieval()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    query = "What are the latest research findings on Abilify's effectiveness for autism in children published after 2016?"
    result = app.invoke({"query": query},config=config)

    print(result["final_answer"])

if __name__ == "__main__":
    main()