from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dlgforge.distributed.provisioning import (
    AttachProvisioner,
    NoopProvisioner,
    _ensure_vllm_available,
)


class DummyResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_noop_provisioner_returns_empty_list() -> None:
    endpoints = asyncio.run(NoopProvisioner().start({}))
    assert endpoints == []


def test_attach_provisioner_requires_endpoints() -> None:
    cfg = {
        "llm": {
            "backend": "vllm_attach",
            "routing": {
                "endpoints": [],
            },
        }
    }
    try:
        asyncio.run(AttachProvisioner().start(cfg))
    except RuntimeError as err:
        assert "llm.backend=vllm_attach" in str(err)
    else:
        raise AssertionError("Expected RuntimeError when no endpoints are configured")


def test_attach_provisioner_validates_models_endpoint() -> None:
    cfg = {
        "llm": {
            "backend": "vllm_attach",
            "routing": {
                "endpoints": [
                    {
                        "name": "node1",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "api_key": "EMPTY",
                    }
                ]
            },
        }
    }

    with patch("dlgforge.distributed.provisioning.requests.get", return_value=DummyResponse(200)) as mocked_get:
        endpoints = asyncio.run(AttachProvisioner().start(cfg))

    assert len(endpoints) == 1
    assert endpoints[0].base_url == "http://127.0.0.1:8000/v1"
    mocked_get.assert_called_once_with("http://127.0.0.1:8000/v1/models", timeout=10.0)


def test_attach_provisioner_fails_on_unhealthy_endpoint() -> None:
    cfg = {
        "llm": {
            "backend": "vllm_attach",
            "routing": {
                "endpoints": [
                    {
                        "name": "node1",
                        "base_url": "http://127.0.0.1:8000",
                        "api_key": "EMPTY",
                    }
                ]
            },
        }
    }

    with patch("dlgforge.distributed.provisioning.requests.get", return_value=DummyResponse(503)):
        try:
            asyncio.run(AttachProvisioner().start(cfg))
        except RuntimeError as err:
            assert "unhealthy" in str(err)
        else:
            raise AssertionError("Expected RuntimeError when endpoint health check fails")


def test_managed_vllm_guard_raises_helpful_error_when_binary_missing() -> None:
    with patch("dlgforge.distributed.provisioning.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Install managed backend deps"):
            _ensure_vllm_available()


def test_managed_vllm_guard_passes_when_binary_available() -> None:
    with patch("dlgforge.distributed.provisioning.shutil.which", return_value="/usr/local/bin/vllm"):
        _ensure_vllm_available()
