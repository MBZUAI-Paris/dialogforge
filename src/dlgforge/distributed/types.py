from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    base_url: str
    api_key: str = ""
    weight: float = 1.0
    max_in_flight: int = 64
    timeout_s: float | None = None

    def to_routing_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "weight": self.weight,
            "max_in_flight": self.max_in_flight,
        }
        if self.timeout_s is not None:
            payload["timeout"] = self.timeout_s
        return payload
