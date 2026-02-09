"""Distributed runtime data types.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class EndpointSpec:
    """Specification for a routed model endpoint.
    
    Args:
        name (str): str value used by this operation.
        base_url (str): str value used by this operation.
        api_key (str): str value used by this operation.
        weight (float): float value used by this operation.
        max_in_flight (int): int value used by this operation.
        timeout_s (float | None): float | None value used by this operation.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.distributed.types import EndpointSpec
        >>> EndpointSpec(...)
    
    """
    name: str
    base_url: str
    api_key: str = ""
    weight: float = 1.0
    max_in_flight: int = 64
    timeout_s: float | None = None

    def to_routing_dict(self) -> Dict[str, Any]:
        """To routing dict.
        
        
        Returns:
            Dict[str, Any]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - Primarily performs in-memory transformations.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.distributed.types import EndpointSpec
            >>> instance = EndpointSpec(...)
            >>> instance.to_routing_dict(...)
        
        """
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
