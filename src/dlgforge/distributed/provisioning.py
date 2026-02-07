from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Protocol, Sequence

import requests

from dlgforge.distributed.ray_runtime import create_vllm_server_actor, import_ray
from dlgforge.distributed.types import EndpointSpec

LOGGER = logging.getLogger("dlgforge.distributed")


class VLLMProvisioner(Protocol):
    async def start(self, cfg: Dict[str, Any]) -> List[EndpointSpec]: ...

    async def stop(self) -> None: ...


class NoopProvisioner:
    async def start(self, cfg: Dict[str, Any]) -> List[EndpointSpec]:
        _ = cfg
        return []

    async def stop(self) -> None:
        return None


class AttachProvisioner:
    async def start(self, cfg: Dict[str, Any]) -> List[EndpointSpec]:
        endpoints = _extract_routing_endpoints(cfg)
        if not endpoints:
            raise RuntimeError(
                "llm.backend=vllm_attach requires llm.routing.endpoints with at least one endpoint."
            )

        await _validate_endpoints(endpoints)
        return endpoints

    async def stop(self) -> None:
        return None


class ManagedRayVLLMProvisioner:
    def __init__(self) -> None:
        self._actors: List[Any] = []
        self._endpoints: List[EndpointSpec] = []
        self._auto_stop: bool = True

    async def start(self, cfg: Dict[str, Any]) -> List[EndpointSpec]:
        _ensure_vllm_available()
        vllm_cfg = (cfg.get("llm", {}) or {}).get("vllm", {}) or {}
        replicas = _as_int(vllm_cfg.get("replicas"), default=1)
        if replicas <= 0:
            raise RuntimeError("llm.vllm.replicas must be >= 1 for vllm_managed backend.")

        model = str(vllm_cfg.get("model") or "").strip()
        served_model_name = str(vllm_cfg.get("served_model_name") or "").strip() or model
        if not model:
            raise RuntimeError("llm.vllm.model is required for llm.backend=vllm_managed.")

        host = str(vllm_cfg.get("host") or "0.0.0.0").strip()
        advertise_host = str(vllm_cfg.get("advertise_host") or "127.0.0.1").strip()
        port_start = _as_int(vllm_cfg.get("port_start"), default=18000)
        num_gpus = float(vllm_cfg.get("num_gpus_per_replica") or 1)
        tensor_parallel = _as_int(vllm_cfg.get("tensor_parallel_size"), default=1)
        gpu_mem = float(vllm_cfg.get("gpu_memory_utilization") or 0.9)
        max_num_seqs = _as_int(vllm_cfg.get("max_num_seqs"), default=256)
        health_timeout_s = float(vllm_cfg.get("health_timeout_s") or 180)
        self._auto_stop = _as_bool(vllm_cfg.get("auto_stop_on_exit"), default=True)

        ray = import_ray()
        actors: List[Any] = []
        endpoints: List[EndpointSpec] = []
        start_refs: List[Any] = []

        for idx in range(replicas):
            port = port_start + idx
            actor = create_vllm_server_actor(num_cpus=1.0, num_gpus=num_gpus)
            actors.append(actor)

            cmd = [
                "vllm",
                "serve",
                model,
                "--host",
                host,
                "--port",
                str(port),
                "--served-model-name",
                served_model_name,
                "--tensor-parallel-size",
                str(tensor_parallel),
                "--gpu-memory-utilization",
                str(gpu_mem),
                "--max-num-seqs",
                str(max_num_seqs),
            ]
            start_refs.append(actor.start_server.remote(cmd, {}))

            endpoints.append(
                EndpointSpec(
                    name=f"managed-vllm-{idx}",
                    base_url=f"http://{advertise_host}:{port}/v1",
                    api_key="EMPTY",
                    weight=1.0,
                    max_in_flight=max(64, max_num_seqs // 2),
                )
            )

        LOGGER.info("[distributed] Starting %s managed vLLM server actor(s)", replicas)
        try:
            await asyncio.to_thread(ray.get, start_refs)
        except Exception as err:
            raise RuntimeError(
                "Failed to start managed vLLM server actors. "
                "Ensure vLLM is installed on Ray worker nodes (`python -m pip install -e \".[vllm]\"`) "
                "and `vllm` is available in PATH."
            ) from err
        await _wait_for_endpoints(endpoints, timeout_s=health_timeout_s)

        self._actors = actors
        self._endpoints = endpoints
        return list(endpoints)

    async def stop(self) -> None:
        if not self._actors:
            return
        if not self._auto_stop:
            return

        ray = import_ray()
        refs = [actor.stop_server.remote() for actor in self._actors]
        try:
            await asyncio.to_thread(ray.get, refs)
        except Exception:
            pass
        self._actors = []


def _extract_routing_endpoints(cfg: Dict[str, Any]) -> List[EndpointSpec]:
    llm_cfg = cfg.get("llm", {}) or {}
    routing_cfg = llm_cfg.get("routing", {}) or {}
    endpoints_raw = routing_cfg.get("endpoints", [])
    if not isinstance(endpoints_raw, list):
        return []

    endpoints: List[EndpointSpec] = []
    for idx, item in enumerate(endpoints_raw):
        if not isinstance(item, dict):
            continue
        base_url = str(item.get("base_url") or "").strip()
        if not base_url:
            continue
        name = str(item.get("name") or f"endpoint-{idx}").strip() or f"endpoint-{idx}"
        api_key = str(item.get("api_key") or "").strip()
        api_key_env = str(item.get("api_key_env") or "").strip()
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env, "")

        endpoints.append(
            EndpointSpec(
                name=name,
                base_url=base_url,
                api_key=api_key,
                weight=float(item.get("weight") or 1.0),
                max_in_flight=_as_int(item.get("max_in_flight"), default=64),
                timeout_s=float(item["timeout_s"]) if item.get("timeout_s") is not None else None,
            )
        )
    return endpoints


async def _validate_endpoints(endpoints: Sequence[EndpointSpec], timeout_s: float = 10.0) -> None:
    checks = [asyncio.to_thread(_check_endpoint_health, ep, timeout_s) for ep in endpoints]
    results = await asyncio.gather(*checks)
    failed = [message for message in results if message]
    if failed:
        joined = "\n".join(failed)
        raise RuntimeError(f"One or more vLLM endpoints are unhealthy:\n{joined}")


async def _wait_for_endpoints(endpoints: Sequence[EndpointSpec], timeout_s: float = 180.0) -> None:
    started = time.time()
    last_error = ""
    while time.time() - started < timeout_s:
        checks = [asyncio.to_thread(_check_endpoint_health, ep, 5.0) for ep in endpoints]
        results = await asyncio.gather(*checks)
        failed = [message for message in results if message]
        if not failed:
            return
        last_error = "\n".join(failed)
        await asyncio.sleep(1.0)

    raise RuntimeError(
        "Timed out waiting for managed vLLM endpoints to become healthy. "
        f"Last errors:\n{last_error}"
    )


def _check_endpoint_health(endpoint: EndpointSpec, timeout_s: float) -> str:
    models_url = _models_url(endpoint.base_url)
    try:
        response = requests.get(models_url, timeout=timeout_s)
        if response.status_code != 200:
            return f"{endpoint.name}: {models_url} -> HTTP {response.status_code}"
    except Exception as err:
        return f"{endpoint.name}: {models_url} -> {err}"
    return ""


def _models_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed + "/models"
    return trimmed + "/v1/models"


def _as_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value


def _as_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _ensure_vllm_available() -> None:
    if shutil.which("vllm"):
        return
    raise RuntimeError(
        "llm.backend=vllm_managed requires `vllm` to be installed and available in PATH. "
        "Install managed backend deps: `python -m pip install -e \".[vllm]\"`."
    )
