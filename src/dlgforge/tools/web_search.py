from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import requests


LOGGER = logging.getLogger("dlgforge.tools")


class SerperWebSearchClient:
    def __init__(self, num_results: int = 5, timeout: int = 30, api_key: str | None = None) -> None:
        self.num_results = num_results
        self.timeout = timeout
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")
        self.session = requests.Session()

    def search(self, query: str) -> Dict[str, Any]:
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
