from __future__ import annotations

from unittest.mock import patch

import pytest

from dlgforge.distributed.bootstrap import _initialize_ray_runtime


class FakeRay:
    def __init__(self, *, initialized: bool = False, first_init_error: Exception | None = None) -> None:
        self._initialized = initialized
        self._first_init_error = first_init_error
        self.init_calls: list[dict[str, object]] = []

    def is_initialized(self) -> bool:
        return self._initialized

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)
        if len(self.init_calls) == 1 and self._first_init_error is not None:
            raise self._first_init_error
        self._initialized = True


def test_initialize_ray_runtime_falls_back_to_local_when_auto_cluster_missing() -> None:
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
    fake_ray = FakeRay(initialized=True)

    with patch("dlgforge.distributed.bootstrap.import_ray", return_value=fake_ray):
        _, started = _initialize_ray_runtime({"ray": {"address": "auto"}})

    assert started is False
    assert fake_ray.init_calls == []
