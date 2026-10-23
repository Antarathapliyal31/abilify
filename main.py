import logging
import os
import uuid

from dotenv import load_dotenv
from langfuse import observe

from retrieval import initialize_retrieval
from graph import app

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@observe()
def main():
    initialize_retrieval()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    query = (
        "What are the latest research findings on Abilify's effectiveness "
        "for autism in children published after 2016?"
    )
    result = app.invoke({"query": query}, config=config)
    print(result.get("final_answer") or result.get("current_answer") or "")


if __name__ == "__main__":
    main()