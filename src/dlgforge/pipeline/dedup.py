from __future__ import annotations

import asyncio
from typing import Iterable, List, Sequence, Set, Tuple


def normalize_question(text: str) -> str:
    return " ".join((text or "").lower().split())


class RunQuestionRegistry:
    def __init__(self, initial_questions: Iterable[str] | None = None) -> None:
        self._lock = asyncio.Lock()
        self._seen: Set[str] = set()
        if initial_questions:
            for question in initial_questions:
                normalized = normalize_question(question)
                if normalized:
                    self._seen.add(normalized)

    async def filter_and_commit(self, candidates: Sequence[Tuple[int, str]]) -> Tuple[Set[int], Set[int]]:
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
