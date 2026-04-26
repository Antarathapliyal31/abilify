from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import asyncio
import traceback
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
    async with streamable_http_client("https://pubmed.caseyjhand.com/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result=await session.call_tool("pubmed_search_articles",{"query":query,"max_results":3})
            if result.content:
                return result.content[0].text
            return "No relevant PubMed articles found."

async def _pubmed_mcp_client_fulltext_async(query:str)->str:
    async with streamable_http_client("https://pubmed.caseyjhand.com/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result=await session.call_tool("pubmed_search_articles",{"query":query,"max_results":3})
            if result.content:
                return result.content[0].text
            return "No relevant PubMed articles found."

def _format_mcp_error(e: BaseException) -> str:
    parts = [f"{type(e).__name__}: {e}"]
    inner = getattr(e, "exceptions", None)
    if inner:
        for sub in inner:
            parts.append(f"  -> {type(sub).__name__}: {sub}")
    print("MCP client traceback:")
    traceback.print_exc()
    return "Error in MCP client: " + " | ".join(parts)

@observe()
def pubmed_mcp_client_search(query:str)->str:
    try:
        return asyncio.run(_pubmed_mcp_client_search_async(query))
    except BaseException as e:
        return _format_mcp_error(e)

@observe()
def pubmed_mcp_client_fulltext(query:str)->str:
    try:
        return asyncio.run(_pubmed_mcp_client_fulltext_async(query))
    except RuntimeError:
        # If event loop already running use this instead
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_pubmed_mcp_client_fulltext_async(query))
    except BaseException as e:
        return _format_mcp_error(e)