from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from dlgforge.distributed.provisioning import (
    AttachProvisioner,
    ManagedRayVLLMProvisioner,
    NoopProvisioner,
    VLLMProvisioner,
)
from dlgforge.distributed.ray_runtime import (
    create_coordinator_actor,
    create_worker_actor,
    import_ray,
)
from dlgforge.distributed.types import EndpointSpec

LOGGER = logging.getLogger("dlgforge.distributed")


class RunBootstrap:
    def __init__(self, config_path: str | Path, cfg: Dict[str, Any]) -> None:
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.cfg = cfg
        self._provisioner: VLLMProvisioner = NoopProvisioner()

    async def run(self, cfg: Dict[str, Any]) -> None:
        _ = cfg
        distributed_cfg = ((self.cfg.get("run", {}) or {}).get("distributed", {}) or {})
        executor = str(distributed_cfg.get("executor") or "ray").strip().lower()
        if executor != "ray":
            raise RuntimeError(
                f"Unsupported run.distributed.executor='{executor}'. Only 'ray' is currently implemented."
            )

        ray, started_ray = _initialize_ray_runtime(self.cfg)

        await _initialize_postgres(self.cfg)

        backend = str((self.cfg.get("llm", {}) or {}).get("backend") or "openai").strip().lower()
        self._provisioner = _select_provisioner(backend)

        endpoints = await self._provisioner.start(self.cfg)
        env_overrides = _build_env_overrides(self.cfg, backend=backend, endpoints=endpoints)

        workers: List[Any] = []
        spawn_cfg = distributed_cfg.get("spawn", {}) if isinstance(distributed_cfg.get("spawn"), dict) else {}
        spawn_workers = bool(spawn_cfg.get("workers", True))
        spawn_coordinator = bool(spawn_cfg.get("coordinator", True))

        ray_actor_cfg = (self.cfg.get("ray", {}) or {}).get("actor", {}) or {}
        worker_replicas_qa = _as_int(ray_actor_cfg.get("replicas_qa"), default=1)
        worker_replicas_complete = _as_int(ray_actor_cfg.get("replicas_complete"), default=1)
        worker_num_cpus = float(ray_actor_cfg.get("num_cpus") or 1.0)
        worker_num_gpus = float(ray_actor_cfg.get("num_gpus") or 0.0)

        if spawn_workers:
            total_workers = max(worker_replicas_qa + worker_replicas_complete, 1)
            for idx in range(total_workers):
                role = "qa" if idx < worker_replicas_qa else "complete"
                workers.append(
                    create_worker_actor(
                        role=role,
                        num_cpus=worker_num_cpus,
                        num_gpus=worker_num_gpus,
                    )
                )
            LOGGER.info("[distributed] Spawned %s worker replica actor(s)", len(workers))

        try:
            if spawn_coordinator:
                coordinator_num_cpus = float(ray_actor_cfg.get("coordinator_num_cpus") or 1.0)
                coordinator = create_coordinator_actor(num_cpus=coordinator_num_cpus)
                result_ref = coordinator.run_generation.remote(self.config_path, env_overrides)
                await asyncio.to_thread(ray.get, result_ref)
            else:
                _run_local_generation(self.config_path, env_overrides)
        finally:
            if workers:
                refs = [worker.shutdown.remote() for worker in workers]
                await asyncio.to_thread(ray.get, refs)
            await self._provisioner.stop()
            if started_ray:
                with _suppress():
                    ray.shutdown()


def run_bootstrap(config_path: str | Path, cfg: Dict[str, Any]) -> None:
    asyncio.run(RunBootstrap(config_path=config_path, cfg=cfg).run(cfg))


def _select_provisioner(backend: str) -> VLLMProvisioner:
    if backend == "openai":
        return NoopProvisioner()
    if backend == "vllm_attach":
        return AttachProvisioner()
    if backend == "vllm_managed":
        return ManagedRayVLLMProvisioner()
    raise RuntimeError(
        f"Unknown llm.backend='{backend}'. Supported values: openai, vllm_attach, vllm_managed."
    )


async def _initialize_postgres(cfg: Dict[str, Any]) -> None:
    store_cfg = cfg.get("store", {}) or {}
    backend = str(store_cfg.get("backend") or "postgres").strip().lower()
    if backend != "postgres":
        raise RuntimeError("run.distributed.enabled currently requires store.backend=postgres.")

    postgres_cfg = store_cfg.get("postgres", {}) if isinstance(store_cfg.get("postgres"), dict) else {}
    dsn = str(postgres_cfg.get("dsn") or os.getenv("DLGFORGE_POSTGRES_DSN") or "").strip()
    if not dsn:
        raise RuntimeError(
            "Missing Postgres DSN. Set store.postgres.dsn or DLGFORGE_POSTGRES_DSN for distributed runtime."
        )

    try:
        import asyncpg
    except Exception as err:  # pragma: no cover - dependency import path
        raise RuntimeError("asyncpg is required for distributed runtime. Install with: `uv pip install asyncpg`.") from err

    conn = await asyncpg.connect(dsn=dsn, timeout=5)
    try:
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


def _build_env_overrides(cfg: Dict[str, Any], backend: str, endpoints: List[EndpointSpec]) -> Dict[str, str]:
    llm_cfg = cfg.get("llm", {}) or {}
    routing_cfg = llm_cfg.get("routing", {}) if isinstance(llm_cfg.get("routing"), dict) else {}
    strategy = str(routing_cfg.get("strategy") or "weighted_least_inflight")

    env: Dict[str, str] = {
        "DLGFORGE_RUN_BOOTSTRAPPED": "1",
        "LLM_BACKEND": backend,
    }

    if endpoints:
        env["LLM_ROUTING_STRATEGY"] = strategy
        env["LLM_ROUTING_ENDPOINTS_JSON"] = json.dumps([endpoint.to_routing_dict() for endpoint in endpoints])

    # If managed vLLM provides a served model name and no model is configured, promote it.
    vllm_cfg = llm_cfg.get("vllm", {}) if isinstance(llm_cfg.get("vllm"), dict) else {}
    served_model_name = str(vllm_cfg.get("served_model_name") or "").strip()
    model = str(llm_cfg.get("model") or "").strip()
    if backend == "vllm_managed" and served_model_name and not model:
        env["LLM_MODEL"] = served_model_name

    return env


def _run_local_generation(config_path: str, env_overrides: Dict[str, str]) -> None:
    from dlgforge.pipeline.runner import run

    with _temp_environ(env_overrides):
        run(config_path)


@contextmanager
def _temp_environ(overrides: Dict[str, str]):
    previous: Dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


class _suppress:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return True


def _as_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _initialize_ray_runtime(cfg: Dict[str, Any]) -> tuple[Any, bool]:
    ray = import_ray()
    if ray.is_initialized():
        return ray, False

    ray_cfg = cfg.get("ray", {}) or {}
    address = str(ray_cfg.get("address") or "auto").strip() or "auto"
    namespace = str(ray_cfg.get("namespace") or "dlgforge").strip() or "dlgforge"
    auto_start_local = _as_bool(ray_cfg.get("auto_start_local"), default=True)
    init_kwargs: Dict[str, Any] = {
        "namespace": namespace,
        "ignore_reinit_error": True,
        "log_to_driver": True,
    }

    try:
        ray.init(address=address, **init_kwargs)
        return ray, True
    except Exception as err:
        if not (_allow_local_fallback(address=address, auto_start_local=auto_start_local, err=err)):
            raise

        LOGGER.warning(
            "[distributed] Ray address='%s' unavailable; starting a local Ray runtime because "
            "ray.auto_start_local=true. Original error: %s",
            address,
            err,
        )
        ray.init(**init_kwargs)
        return ray, True


def _allow_local_fallback(address: str, auto_start_local: bool, err: Exception) -> bool:
    if not auto_start_local:
        return False
    if address.lower() != "auto":
        return False

    message = str(err).lower()
    return (
        "could not find any running ray instance" in message
        or "please specify the one to connect to" in message
    )


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}
