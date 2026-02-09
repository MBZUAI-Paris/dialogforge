"""
Test module for DialogForge behavior validation.

Main flows:
- Defines unit/integration test cases that assert expected runtime behavior.

Expected usage:
- Run with `pytest`; import helpers from production modules as needed.
"""

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
    """
    Encapsulate DummyResponse behavior.

    Args:
        None.

    Returns:
        DummyResponse: Instance of `DummyResponse`.

    Raises:
        Exception: Propagates constructor or lifecycle runtime errors when applicable.

    Side Effects:
        - May initialize state used by test/script execution.

    Preconditions/Invariant:
        Use through documented public methods in this class.

    Example:
        >>> # Instantiate `DummyResponse` and call its public methods.

    Notes/Assumptions:
        Public methods: none (container type).
    """
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_noop_provisioner_returns_empty_list() -> None:
    """
    Test noop provisioner returns empty list.

    Args:
        None.

    Returns:
        None: Return value produced by `test_noop_provisioner_returns_empty_list`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_provisioning.py` for concrete usage of `test_noop_provisioner_returns_empty_list`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    endpoints = asyncio.run(NoopProvisioner().start({}))
    assert endpoints == []


def test_attach_provisioner_requires_endpoints() -> None:
    """
    Test attach provisioner requires endpoints.

    Args:
        None.

    Returns:
        None: Return value produced by `test_attach_provisioner_requires_endpoints`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_provisioning.py` for concrete usage of `test_attach_provisioner_requires_endpoints`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
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
    """
    Test attach provisioner validates models endpoint.

    Args:
        None.

    Returns:
        None: Return value produced by `test_attach_provisioner_validates_models_endpoint`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_provisioning.py` for concrete usage of `test_attach_provisioner_validates_models_endpoint`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
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
    """
    Test attach provisioner fails on unhealthy endpoint.

    Args:
        None.

    Returns:
        None: Return value produced by `test_attach_provisioner_fails_on_unhealthy_endpoint`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_provisioning.py` for concrete usage of `test_attach_provisioner_fails_on_unhealthy_endpoint`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
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
    """
    Test managed vllm guard raises helpful error when binary missing.

    Args:
        None.

    Returns:
        None: Return value produced by `test_managed_vllm_guard_raises_helpful_error_when_binary_missing`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_provisioning.py` for concrete usage of `test_managed_vllm_guard_raises_helpful_error_when_binary_missing`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    with patch("dlgforge.distributed.provisioning.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="Install managed backend deps"):
            _ensure_vllm_available()


def test_managed_vllm_guard_passes_when_binary_available() -> None:
    """
    Test managed vllm guard passes when binary available.

    Args:
        None.

    Returns:
        None: Return value produced by `test_managed_vllm_guard_passes_when_binary_available`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_provisioning.py` for concrete usage of `test_managed_vllm_guard_passes_when_binary_available`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    with patch("dlgforge.distributed.provisioning.shutil.which", return_value="/usr/local/bin/vllm"):
        _ensure_vllm_available()
