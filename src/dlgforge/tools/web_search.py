"""Serper web-search integration helper.

"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import requests

LOGGER = logging.getLogger("dlgforge.tools")

class SerperWebSearchClient:
    """Serper web search client.
    
    Args:
        num_results (int): int value used by this operation.
        timeout (int): int value used by this operation.
        api_key (str | None): str | None value used by this operation.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.tools.web_search import SerperWebSearchClient
        >>> SerperWebSearchClient(...)
    
    """
    def __init__(self, num_results: int = 5, timeout: int = 30, api_key: str | None = None) -> None:
        self.num_results = num_results
        self.timeout = timeout
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")
        self.session = requests.Session()

    def search(self, query: str) -> Dict[str, Any]:
        """Execute a web search query.
        
        Args:
            query (str): Input text.
        
        Returns:
            Dict[str, Any]: Value produced by this API.
        
        Raises:
            RuntimeError: Raised when validation or runtime requirements are not met.
        
        Side Effects / I/O:
            - May perform network, model, or distributed runtime operations.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.web_search import SerperWebSearchClient
            >>> instance = SerperWebSearchClient(...)
            >>> instance.search(...)
        
        """
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY is not set.")

        start = time.perf_counter()
        LOGGER.info(f"[tools] web_search query_len={len(query)}")
        payload = {"q": query, "num": self.num_results}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        response = self.session.post(
            "https://google.serper.dev/search",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        organic_results: List[Dict[str, Any]] = data.get("organic", []) or []
        results = []
        for item in organic_results[: self.num_results]:
            results.append(
                {
                    "title": item.get("title", "No title"),
                    "link": item.get("link", "No link"),
                    "snippet": item.get("snippet", "No snippet provided."),
                }
            )

        rendered = "\n".join(
            f"- {row['title']} ({row['link']}): {row['snippet']}" for row in results
        )
        if not rendered:
            rendered = "No web results were returned for this query."

        LOGGER.info(
            f"[tools] web_search results={len(results)} elapsed={time.perf_counter() - start:.2f}s"
        )
        return {
            "query": query,
            "results": results,
            "rendered": rendered,
        }
