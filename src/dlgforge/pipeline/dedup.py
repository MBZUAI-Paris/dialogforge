"""Question normalization and dedup registry helpers.

"""

from __future__ import annotations

import asyncio
from typing import Iterable, List, Sequence, Set, Tuple

def normalize_question(text: str) -> str:
    """Normalize question.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.pipeline.dedup import normalize_question
        >>> normalize_question(...)
    
    """
    return " ".join((text or "").lower().split())

class RunQuestionRegistry:
    """Helper for runquestionregistry.
    
    Args:
        initial_questions (Iterable[str] | None): Iterable[str] | None value used by this operation.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - Primarily performs in-memory transformations.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.pipeline.dedup import RunQuestionRegistry
        >>> RunQuestionRegistry(...)
    
    """
    def __init__(self, initial_questions: Iterable[str] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._seen: Set[str] = set()
        if initial_questions:
            for question in initial_questions:
                normalized = normalize_question(question)
                if normalized:
                    self._seen.add(normalized)

    async def filter_and_commit(self, candidates: Sequence[Tuple[int, str]]) -> Tuple[Set[int], Set[int]]:
        """Filter and commit.
        
        Args:
            candidates (Sequence[Tuple[int, str]]): Sequence[Tuple[int, str]] value used by this operation.
        
        Returns:
            Tuple[Set[int], Set[int]]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - Primarily performs in-memory transformations.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.pipeline.dedup import RunQuestionRegistry
            >>> instance = RunQuestionRegistry(...)
            >>> instance.filter_and_commit(...)
        
        """
        accepted: Set[int] = set()
        rejected: Set[int] = set()
        pending: List[Tuple[int, str]] = sorted(candidates, key=lambda item: item[0])

        async with self._lock:
            seen_in_batch: Set[str] = set()
            to_commit: Set[str] = set()
            for conversation_index, question in pending:
                normalized = normalize_question(question)
                if not normalized:
                    rejected.add(conversation_index)
                    continue
                if normalized in seen_in_batch or normalized in self._seen:
                    rejected.add(conversation_index)
                    continue

                seen_in_batch.add(normalized)
                to_commit.add(normalized)
                accepted.add(conversation_index)

            self._seen.update(to_commit)

        return accepted, rejected
