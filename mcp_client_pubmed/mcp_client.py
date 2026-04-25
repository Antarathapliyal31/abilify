from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import asyncio
"""streamable_http_client:
Handles the HTTP connection layer
Connects to the server URL
Returns two things:
  read  — channel to RECEIVE data from server
  write — channel to SEND data to server"""
import os
from dotenv import load_dotenv
from langfuse import observe
load_dotenv()

async def _pubmed_mcp_client_search_async(query:str)->str:
    async with streamable_http_client("https://pubmed.caseyjhand.com/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result=await session.call_tool("pubmed_search_articles",{"query":query,"max_results":3})
            if result.content:
                return result.content[0].text
            return "No relevant PubMed articles found."

async def _pubmed_mcp_client_fulltext_async(query:str)->str:
    async with streamable_http_client("https://pubmed.caseyjhand.com/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result=await session.call_tool("pubmed_search_articles",{"query":query,"max_results":3})
            if result.content:
                return result.content[0].text
            return "No relevant PubMed articles found."

@observe()    
def pubmed_mcp_client_search(query:str)->str:
    try:
        return asyncio.run(_pubmed_mcp_client_search_async(query))
    except Exception as e:
        return f"Error in MCP client: {e}"
@observe()
def pubmed_mcp_client_fulltext(query:str)->str:
    try:
        return asyncio.run(_pubmed_mcp_client_fulltext_async(query))
    except Exception as e:
        return f"Error in MCP client: {e}"