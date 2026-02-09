"""Ray actor construction and runtime helpers.

"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import suppress
from typing import Any, Dict, List

def import_ray() -> Any:
    """Import ray.
    
    
    Returns:
        Any: Value produced by this API.
    
    Raises:
        RuntimeError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.distributed.ray_runtime import import_ray
        >>> import_ray(...)
    
    """
    try:
        import ray  # type: ignore
    except Exception as err:  # pragma: no cover - import errors are environment-specific
        raise RuntimeError(
            "Ray is required for distributed executor mode. Install with: `uv pip install ray`."
        ) from err
    return ray

def create_worker_actor(role: str, num_cpus: float = 1.0, num_gpus: float = 0.0) -> Any:
    """Create worker actor.
    
    Args:
        role (str): str value used by this operation.
        num_cpus (float): float value used by this operation.
        num_gpus (float): float value used by this operation.
    
    Returns:
        Any: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.distributed.ray_runtime import create_worker_actor
        >>> create_worker_actor(...)
    
    """
    ray = import_ray()

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class WorkerReplicaActor:
        def __init__(self, worker_role: str) -> None:
            self._worker_role = worker_role
            self._started_at = time.time()

        def ping(self) -> Dict[str, Any]:
            return {
                "role": self._worker_role,
                "uptime_s": time.time() - self._started_at,
            }

        def shutdown(self) -> bool:
            return True

    return WorkerReplicaActor.remote(role)

def create_coordinator_actor(num_cpus: float = 1.0) -> Any:
    """Create coordinator actor.
    
    Args:
        num_cpus (float): float value used by this operation.
    
    Returns:
        Any: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.distributed.ray_runtime import create_coordinator_actor
        >>> create_coordinator_actor(...)
    
    """
    ray = import_ray()

    @ray.remote(num_cpus=num_cpus)
    class CoordinatorActor:
        def run_generation(self, config_path: str, env_overrides: Dict[str, str]) -> Dict[str, Any]:
            previous: Dict[str, str | None] = {}
            try:
                for key, value in env_overrides.items():
                    previous[key] = os.environ.get(key)
                    os.environ[key] = value
                os.environ["DLGFORGE_RUN_BOOTSTRAPPED"] = "1"

                from dlgforge.pipeline.runner import run

                run(config_path)
                return {"status": "completed"}
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old

    return CoordinatorActor.remote()

def create_vllm_server_actor(num_cpus: float = 1.0, num_gpus: float = 1.0) -> Any:
    """Create vllm server actor.
    
    Args:
        num_cpus (float): float value used by this operation.
        num_gpus (float): float value used by this operation.
    
    Returns:
        Any: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May perform network, model, or distributed runtime operations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.distributed.ray_runtime import create_vllm_server_actor
        >>> create_vllm_server_actor(...)
    
    """
    ray = import_ray()

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class VLLMServerActor:
        def __init__(self) -> None:
            self._proc: subprocess.Popen[str] | None = None
            self._cmd: List[str] = []

        def start_server(self, command: List[str], env_overrides: Dict[str, str] | None = None) -> Dict[str, Any]:
            if self._proc and self._proc.poll() is None:
                return {"status": "already_running", "pid": self._proc.pid, "command": self._cmd}
            if not shutil.which("vllm"):
                raise RuntimeError(
                    "Managed vLLM mode requires `vllm` on Ray worker nodes. "
                    "Install with: `python -m pip install -e \".[vllm]\"`."
                )

            env = os.environ.copy()
            if env_overrides:
                env.update(env_overrides)

            self._cmd = list(command)
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
            )
            return {"status": "started", "pid": self._proc.pid, "command": self._cmd}

        def health(self) -> Dict[str, Any]:
            if not self._proc:
                return {"running": False, "reason": "not_started"}
            return {
                "running": self._proc.poll() is None,
                "pid": self._proc.pid,
                "returncode": self._proc.poll(),
                "command": self._cmd,
            }

        def stop_server(self) -> Dict[str, Any]:
            if not self._proc:
                return {"status": "not_started"}
            if self._proc.poll() is not None:
                return {"status": "already_stopped", "returncode": self._proc.returncode}

            self._proc.terminate()
            with suppress(Exception):
                self._proc.wait(timeout=10)
            if self._proc.poll() is None:
                with suppress(Exception):
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            return {"status": "stopped", "returncode": self._proc.returncode}

    return VLLMServerActor.remote()
