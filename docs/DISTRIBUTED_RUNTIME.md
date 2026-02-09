# Distributed Runtime

## Overview
Distributed execution is enabled by `run.distributed.enabled=true`. The runtime uses Ray actors for orchestration and requires Postgres for distributed state prerequisites.

## Bootstrap lifecycle
Entry path:
- `src/dlgforge/pipeline/runner.py` detects distributed mode.
- `src/dlgforge/distributed/bootstrap.py` executes bootstrap.

Bootstrap stages:
1. Initialize Ray runtime (`ray.address` with optional local fallback for `auto`).
2. Validate Postgres DSN and connectivity (`SELECT 1`).
3. Select provisioner based on `llm.backend`.
4. Optionally spawn worker and coordinator actors.
5. Execute generation under bootstrap env overrides.
6. Stop provisioned resources and optionally shutdown Ray started by bootstrap.

## Backend behavior
- `openai`: no vLLM provisioning; routed endpoint overrides are not required.
- `vllm_attach`: validates configured endpoints (`/v1/models`) before run.
- `vllm_managed`: starts vLLM server actors on Ray workers and waits until healthy.

## Required prerequisites
- `store.backend=postgres` for distributed path.
- Postgres DSN provided by `store.postgres.dsn` or `DLGFORGE_POSTGRES_DSN`.
- Ray installed and reachable.
- For managed vLLM: `vllm` binary available on worker nodes.

## Runtime env overrides injected by bootstrap
Typical overrides set for the coordinator execution environment:
- `DLGFORGE_RUN_BOOTSTRAPPED=1`
- `LLM_BACKEND=<backend>`
- `LLM_ROUTING_STRATEGY=<strategy>` (when endpoints exist)
- `LLM_ROUTING_ENDPOINTS_JSON=<json>` (when endpoints exist)
- `LLM_MODEL=<served_model_name>` in managed mode when model is otherwise unset

## Failure modes
Common bootstrap failures:
- Missing or invalid Postgres DSN.
- Ray cluster unavailable and local fallback disabled.
- `vllm_attach` endpoints unhealthy.
- `vllm_managed` missing `vllm` binary or failing server startup.

## Operational recommendations
- Keep `ray.auto_start_local=true` for laptop/dev convenience.
- In production, use explicit Ray addresses and monitored Postgres.
- Validate endpoint health out-of-band before long runs.
- Keep `llm.vllm.served_model_name` aligned with agent model names in managed mode.
