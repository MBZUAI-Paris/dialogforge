"""
Test module for DialogForge behavior validation.

Main flows:
- Defines unit/integration test cases that assert expected runtime behavior.

Expected usage:
- Run with `pytest`; import helpers from production modules as needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dlgforge.distributed.bootstrap import _initialize_ray_runtime


class FakeRay:
    """
    Encapsulate FakeRay behavior.

    Args:
        None.

    Returns:
        FakeRay: Instance of `FakeRay`.

    Raises:
        Exception: Propagates constructor or lifecycle runtime errors when applicable.

    Side Effects:
        - May initialize state used by test/script execution.

    Preconditions/Invariant:
        Use through documented public methods in this class.

    Example:
        >>> # Instantiate `FakeRay` and call its public methods.

    Notes/Assumptions:
        Public methods: is_initialized, init.
    """
    def __init__(self, *, initialized: bool = False, first_init_error: Exception | None = None) -> None:
        self._initialized = initialized
        self._first_init_error = first_init_error
        self.init_calls: list[dict[str, object]] = []

    def is_initialized(self) -> bool:
        """
        Handle is initialized.

        Args:
            None.

        Returns:
            bool: Return value produced by `is_initialized`.

        Raises:
            Exception: Propagates assertion, validation, or runtime errors from executed code paths.

        Side Effects:
            - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

        Preconditions/Invariant:
            Callers should provide values compatible with the expected schema and test/script context.

        Example:
            >>> # See `tests/test_distributed_bootstrap.py` for concrete usage of `is_initialized`.

        Notes/Assumptions:
            This callable is used by pytest discovery and assertions.
        """
        return self._initialized

    def init(self, **kwargs: object) -> None:
        """
        Handle init.

        Args:
            **kwargs (object): Parameter consumed by `init`.

        Returns:
            None: Return value produced by `init`.

        Raises:
            Exception: Propagates assertion, validation, or runtime errors from executed code paths.

        Side Effects:
            - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

        Preconditions/Invariant:
            Callers should provide values compatible with the expected schema and test/script context.

        Example:
            >>> # See `tests/test_distributed_bootstrap.py` for concrete usage of `init`.

        Notes/Assumptions:
            This callable is used by pytest discovery and assertions.
        """
        self.init_calls.append(kwargs)
        if len(self.init_calls) == 1 and self._first_init_error is not None:
            raise self._first_init_error
        self._initialized = True


def test_initialize_ray_runtime_falls_back_to_local_when_auto_cluster_missing() -> None:
    """
    Test initialize ray runtime falls back to local when auto cluster missing.

    Args:
        None.

    Returns:
        None: Return value produced by `test_initialize_ray_runtime_falls_back_to_local_when_auto_cluster_missing`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_bootstrap.py` for concrete usage of `test_initialize_ray_runtime_falls_back_to_local_when_auto_cluster_missing`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    fake_ray = FakeRay(
        initialized=False,
        first_init_error=RuntimeError(
            "Could not find any running Ray instance. Please specify the one to connect to."
        ),
    )
    cfg = {
        "ray": {
            "address": "auto",
            "namespace": "dlgforge",
            "auto_start_local": True,
        }
    }

    with patch("dlgforge.distributed.bootstrap.import_ray", return_value=fake_ray):
        _, started = _initialize_ray_runtime(cfg)

    assert started is True
    assert len(fake_ray.init_calls) == 2
    assert fake_ray.init_calls[0]["address"] == "auto"
    assert "address" not in fake_ray.init_calls[1]


def test_initialize_ray_runtime_does_not_fallback_when_disabled() -> None:
    """
    Test initialize ray runtime does not fallback when disabled.

    Args:
        None.

    Returns:
        None: Return value produced by `test_initialize_ray_runtime_does_not_fallback_when_disabled`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_bootstrap.py` for concrete usage of `test_initialize_ray_runtime_does_not_fallback_when_disabled`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    fake_ray = FakeRay(
        initialized=False,
        first_init_error=RuntimeError(
            "Could not find any running Ray instance. Please specify the one to connect to."
        ),
    )
    cfg = {
        "ray": {
            "address": "auto",
            "namespace": "dlgforge",
            "auto_start_local": False,
        }
    }

    with patch("dlgforge.distributed.bootstrap.import_ray", return_value=fake_ray):
        with pytest.raises(RuntimeError, match="Could not find any running Ray instance"):
            _initialize_ray_runtime(cfg)

    assert len(fake_ray.init_calls) == 1


def test_initialize_ray_runtime_does_not_fallback_for_explicit_address() -> None:
    """
    Test initialize ray runtime does not fallback for explicit address.

    Args:
        None.

    Returns:
        None: Return value produced by `test_initialize_ray_runtime_does_not_fallback_for_explicit_address`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_bootstrap.py` for concrete usage of `test_initialize_ray_runtime_does_not_fallback_for_explicit_address`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    fake_ray = FakeRay(
        initialized=False,
        first_init_error=RuntimeError(
            "Could not find any running Ray instance. Please specify the one to connect to."
        ),
    )
    cfg = {
        "ray": {
            "address": "ray://10.0.0.1:10001",
            "namespace": "dlgforge",
            "auto_start_local": True,
        }
    }

    with patch("dlgforge.distributed.bootstrap.import_ray", return_value=fake_ray):
        with pytest.raises(RuntimeError, match="Could not find any running Ray instance"):
            _initialize_ray_runtime(cfg)

    assert len(fake_ray.init_calls) == 1


def test_initialize_ray_runtime_noop_when_ray_already_initialized() -> None:
    """
    Test initialize ray runtime noop when ray already initialized.

    Args:
        None.

    Returns:
        None: Return value produced by `test_initialize_ray_runtime_noop_when_ray_already_initialized`.

    Raises:
        Exception: Propagates assertion, validation, or runtime errors from executed code paths.

    Side Effects:
        - May read/write local test artifacts, parse CLI args, or invoke runtime utilities.

    Preconditions/Invariant:
        Callers should provide values compatible with the expected schema and test/script context.

    Example:
        >>> # See `tests/test_distributed_bootstrap.py` for concrete usage of `test_initialize_ray_runtime_noop_when_ray_already_initialized`.

    Notes/Assumptions:
        This callable is used by pytest discovery and assertions.
    """
    fake_ray = FakeRay(initialized=True)

    with patch("dlgforge.distributed.bootstrap.import_ray", return_value=fake_ray):
        _, started = _initialize_ray_runtime({"ray": {"address": "auto"}})

    assert started is False
    assert fake_ray.init_calls == []
