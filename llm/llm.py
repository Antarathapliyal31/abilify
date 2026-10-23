import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

# Configurable via .env — pick a Bedrock model + region without code changes.
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")

llm = ChatBedrock(model_id=BEDROCK_MODEL_ID, region_name=BEDROCK_REGION)
