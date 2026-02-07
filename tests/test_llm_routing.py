from __future__ import annotations

import os

from dlgforge.llm.client import OpenAIModelClient


def test_settings_with_routed_endpoint_overrides_base_url_and_api_key() -> None:
    client = OpenAIModelClient()
    settings = {
        "model": "demo-model",
        "api_key": "",
        "routing": {
            "endpoints": [
                {
                    "name": "node-a",
                    "base_url": "http://127.0.0.1:8001/v1",
                    "api_key": "EMPTY",
                }
            ]
        },
    }

    routed, endpoint_name, endpoint_key = client._settings_with_routed_endpoint(settings)
    assert routed["base_url"] == "http://127.0.0.1:8001/v1"
    assert routed["api_key"] == "EMPTY"
    assert endpoint_name == "node-a"
    assert endpoint_key


def test_settings_with_routed_endpoint_reads_api_key_env() -> None:
    client = OpenAIModelClient()
    previous = os.environ.get("TEST_ROUTING_API_KEY")
    try:
        os.environ["TEST_ROUTING_API_KEY"] = "token-123"
        settings = {
            "model": "demo-model",
            "routing": {
                "endpoints": [
                    {
                        "name": "node-b",
                        "base_url": "http://127.0.0.1:8002/v1",
                        "api_key_env": "TEST_ROUTING_API_KEY",
                    }
                ]
            },
        }

        routed, endpoint_name, _ = client._settings_with_routed_endpoint(settings)
        assert routed["api_key"] == "token-123"
        assert endpoint_name == "node-b"
    finally:
        if previous is None:
            os.environ.pop("TEST_ROUTING_API_KEY", None)
        else:
            os.environ["TEST_ROUTING_API_KEY"] = previous
