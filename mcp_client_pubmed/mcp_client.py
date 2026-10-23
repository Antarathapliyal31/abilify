import asyncio
import logging
import os

from dotenv import load_dotenv
from langfuse import observe
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()
logger = logging.getLogger(__name__)

PUBMED_MCP_URL = os.getenv("PUBMED_MCP_URL", "https://pubmed.caseyjhand.com/mcp")

async def _pubmed_mcp_client_search_async(query: str) -> str:
    async with streamable_http_client(PUBMED_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "pubmed_search_articles", {"query": query, "max_results": 3}
            )
            if result.content:
                return result.content[0].text
            return "No relevant PubMed articles found."


async def _pubmed_mcp_client_fulltext_async(pmid: str) -> str:
    async with streamable_http_client(PUBMED_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("pubmed_fetch_fulltext", {"pmid": pmid})
            if result.content:
                return result.content[0].text
            return "No relevant PubMed articles found."


def _format_mcp_error(e: BaseException) -> str:
    parts = [f"{type(e).__name__}: {e}"]
    for sub in getattr(e, "exceptions", None) or ():
        parts.append(f"  -> {type(sub).__name__}: {sub}")
    logger.exception("PubMed MCP call failed")
    return "Error in MCP client: " + " | ".join(parts)


@observe()
def pubmed_mcp_client_search(query: str) -> str:
    try:
        return asyncio.run(_pubmed_mcp_client_search_async(query))
    except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
        return _format_mcp_error(e)


@observe()
def pubmed_mcp_client_fulltext(pmid: str) -> str:
    try:
        return asyncio.run(_pubmed_mcp_client_fulltext_async(pmid))
    except RuntimeError:
        # Called from inside a running event loop (e.g. Streamlit).
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_pubmed_mcp_client_fulltext_async(pmid))
    except (ConnectionError, TimeoutError, OSError) as e:
        return _format_mcp_error(e)