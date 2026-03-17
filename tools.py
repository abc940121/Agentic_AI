import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient()

def search_tool(query: str) -> str:
    try:
        result=client.qna_search(query)
        return result
    except Exception as e:
        return f"Search failed: {e}"