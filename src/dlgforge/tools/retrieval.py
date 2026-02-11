"""Knowledge retrieval and vector-store utilities.

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

LOGGER = logging.getLogger("dlgforge.retrieval")

def _default_device() -> Optional[str]:
    with suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            return "cuda"
    return None

def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default

class KnowledgeVectorStore:
    """Vector-store wrapper for indexing and retrieving knowledge passages.
    
    Args:
        knowledge_dir (Path): Path value used by this operation.
        collection_name (str): str value used by this operation.
        chunk_size (int): int value used by this operation.
        overlap (int): int value used by this operation.
        embedding_model_name (Optional[str]): Optional[str] value used by this operation.
        persist_dir (Optional[Path]): Optional[Path] value used by this operation.
        rebuild_index (bool): bool value used by this operation.
        skip_if_unchanged (bool): bool value used by this operation.
    
    Raises:
        Exception: Construction may raise when required dependencies or inputs are invalid.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Instantiate and use through documented public methods.
    
    Examples:
        >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
        >>> KnowledgeVectorStore(...)
    
    """

    def __init__(
        self,
        knowledge_dir: Path,
        collection_name: str = "dlgforge_knowledge",
        chunk_size: int = 750,
        overlap: int = 150,
        embedding_model_name: Optional[str] = None,
        persist_dir: Optional[Path] = None,
        rebuild_index: bool = False,
        skip_if_unchanged: bool = True,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.persist_dir = persist_dir
        self.rebuild_index = rebuild_index
        self.skip_if_unchanged = skip_if_unchanged
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        model_name = embedding_model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_model_name = model_name
        embedding_start = time.perf_counter()
        LOGGER.info(f"[vector-db] Initializing embedding model={self.embedding_model_name}")
        self.embedding_function = _build_embedding_function(self.embedding_model_name)
        LOGGER.info(f"[vector-db] Embedding model ready in {time.perf_counter() - embedding_start:.2f}s")
        LOGGER.info(f"[vector-db] Using embedding model: {model_name}")

        self.client = self._build_client()
        if self.rebuild_index:
            self._reset_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

        self._chunk_inventory: List[Dict[str, Any]] = []
        self._load_documents()

    def _build_client(self):
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=str(self.persist_dir))
        return chromadb.Client()

    def _reset_collection(self) -> None:
        with suppress(Exception):
            self.client.delete_collection(name=self.collection_name)

    def _load_documents(self) -> None:
        load_start = time.perf_counter()
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        skip_upsert = False
        fingerprint = None
        indexed_files = 0

        source_files = [
            path
            for path in sorted(self.knowledge_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
        ]
        LOGGER.info(
            f"[vector-db] Index check: files={len(source_files)} "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )

        if self.persist_dir and self.skip_if_unchanged:
            fingerprint = self._compute_fingerprint()
            if self._fingerprint_matches(fingerprint):
                if self._load_chunk_cache():
                    LOGGER.info(
                        "[vector-db] Knowledge index unchanged. Loaded cached chunks. "
                        f"chunks={len(self._chunk_inventory)} elapsed={time.perf_counter() - load_start:.2f}s"
                    )
                    return
                if self.collection.count() > 0:
                    skip_upsert = True

        for idx_file, path in enumerate(source_files, start=1):
            text = self._read_file_text(path)
            if not text:
                continue
            indexed_files += 1

            source = str(path.relative_to(self.knowledge_dir))
            for idx, (chunk, start, end) in enumerate(self._chunk_text(text)):
                metadata = {
                    "source": source,
                    "chunk_index": idx,
                    "char_start": start,
                    "char_end": end,
                }
                chunk_id = self._build_chunk_id(path, idx)
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(chunk_id)
                self._chunk_inventory.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": metadata,
                    }
                )
            if idx_file % 25 == 0:
                LOGGER.info(f"[vector-db] Indexed {idx_file}/{len(source_files)} files...")

        if documents and not skip_upsert:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

        if self.persist_dir:
            if fingerprint is None:
                fingerprint = self._compute_fingerprint()
            self._persist_fingerprint(fingerprint)
            self._persist_chunk_cache()

        LOGGER.info(
            f"[vector-db] Index ready: files_with_text={indexed_files}, chunks={len(self._chunk_inventory)}, "
            f"upserted={len(documents) if not skip_upsert else 0}, elapsed={time.perf_counter() - load_start:.2f}s"
        )

    def _read_file_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            try:
                return path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                return path.read_text(encoding="latin-1", errors="ignore").strip()

        if suffix == ".pdf":
            try:
                reader = PdfReader(str(path))
                text = "\n".join(self._extract_page_text(page) for page in reader.pages)
                return text.strip()
            except Exception:
                return ""

        return ""

    def _extract_page_text(self, page) -> str:
        try:
            layout_text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            layout_text = ""
        plain_text = page.extract_text() or ""
        return _choose_best_text(layout_text, plain_text)

    def _chunk_text(self, text: str) -> Sequence[Tuple[str, int, int]]:
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                yield chunk, start, end
            if end == text_length:
                break
            start = max(0, end - self.overlap)

    def _build_chunk_id(self, path: Path, idx: int) -> str:
        digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
        return f"{digest}-{idx}"

    def _fingerprint_path(self) -> Optional[Path]:
        if not self.persist_dir:
            return None
        return self.persist_dir / "knowledge_fingerprint.json"

    def _chunk_cache_path(self) -> Optional[Path]:
        if not self.persist_dir:
            return None
        return self.persist_dir / "chunk_inventory.jsonl"

    def _compute_fingerprint(self) -> str:
        files = []
        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": str(path.relative_to(self.knowledge_dir)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

        payload = {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "files": files,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _fingerprint_matches(self, fingerprint: str) -> bool:
        path = self._fingerprint_path()
        if not path or not path.exists():
            return False
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            return stored.get("fingerprint") == fingerprint
        except Exception:
            return False

    def _persist_fingerprint(self, fingerprint: str) -> None:
        path = self._fingerprint_path()
        if not path:
            return
        payload = {"fingerprint": fingerprint}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_chunk_cache(self) -> None:
        path = self._chunk_cache_path()
        if not path:
            return
        with path.open("w", encoding="utf-8") as handle:
            for item in self._chunk_inventory:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _load_chunk_cache(self) -> bool:
        path = self._chunk_cache_path()
        if not path or not path.exists():
            return False
        try:
            items = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                items.append(json.loads(line))
            if items:
                self._chunk_inventory = items
                return True
        except Exception:
            return False
        return False

    def similarity_search(self, query: str, k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Similarity search.
        
        Args:
            query (str): Input text.
            k (int): Numeric control value for processing behavior.
        
        Returns:
            List[Tuple[str, Dict[str, Any]]]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - May read from or write to local filesystem artifacts.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
            >>> instance = KnowledgeVectorStore(...)
            >>> instance.similarity_search(...)
        
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return list(zip(documents, metadatas))

    def similarity_search_with_ids(self, query: str, k: int) -> List[Tuple[str, Dict[str, Any], str]]:
        """Similarity search with ids.
        
        Args:
            query (str): Input text.
            k (int): Numeric control value for processing behavior.
        
        Returns:
            List[Tuple[str, Dict[str, Any], str]]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - May read from or write to local filesystem artifacts.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
            >>> instance = KnowledgeVectorStore(...)
            >>> instance.similarity_search_with_ids(...)
        
        """
        results = self.collection.query(query_texts=[query], n_results=k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return list(zip(documents, metadatas, ids))

    def list_sources(self) -> List[str]:
        """List sources.
        
        
        Returns:
            List[str]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - May read from or write to local filesystem artifacts.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
            >>> instance = KnowledgeVectorStore(...)
            >>> instance.list_sources(...)
        
        """
        sources = {item["metadata"].get("source") for item in self._chunk_inventory}
        return sorted([source for source in sources if source])

    def source_chunk_counts(self) -> Dict[str, int]:
        """Source chunk counts.
        
        
        Returns:
            Dict[str, int]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - May read from or write to local filesystem artifacts.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
            >>> instance = KnowledgeVectorStore(...)
            >>> instance.source_chunk_counts(...)
        
        """
        counts: Dict[str, int] = {}
        for item in self._chunk_inventory:
            source = item["metadata"].get("source")
            if not source:
                continue
            counts[source] = counts.get(source, 0) + 1
        return counts

    def random_samples(
        self,
        n: int,
        exclude_ids: Optional[set[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """Random samples.
        
        Args:
            n (int): Numeric control value for processing behavior.
            exclude_ids (Optional[set[str]]): Optional[set[str]] value used by this operation.
            rng (Optional[random.Random]): Optional[random.Random] value used by this operation.
        
        Returns:
            List[Tuple[str, Dict[str, Any], str]]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - May read from or write to local filesystem artifacts.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
            >>> instance = KnowledgeVectorStore(...)
            >>> instance.random_samples(...)
        
        """
        if not self._chunk_inventory:
            return []
        exclude_ids = exclude_ids or set()
        pool = [item for item in self._chunk_inventory if item["id"] not in exclude_ids]
        if not pool:
            return []
        rng = rng or random.Random()
        rng.shuffle(pool)
        selected = pool[:n]
        return [(item["text"], item["metadata"], item["id"]) for item in selected]

    def sample_by_sources(
        self,
        sources: set[str],
        n: int,
        exclude_ids: Optional[set[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """Sample by sources.
        
        Args:
            sources (set[str]): set[str] value used by this operation.
            n (int): Numeric control value for processing behavior.
            exclude_ids (Optional[set[str]]): Optional[set[str]] value used by this operation.
            rng (Optional[random.Random]): Optional[random.Random] value used by this operation.
        
        Returns:
            List[Tuple[str, Dict[str, Any], str]]: Value produced by this API.
        
        Raises:
            Exception: Propagates unexpected runtime errors from downstream calls.
        
        Side Effects / I/O:
            - May read from or write to local filesystem artifacts.
        
        Preconditions / Invariants:
            - Callers should provide arguments matching annotated types and expected data contracts.
        
        Examples:
            >>> from dlgforge.tools.retrieval import KnowledgeVectorStore
            >>> instance = KnowledgeVectorStore(...)
            >>> instance.sample_by_sources(...)
        
        """
        if not self._chunk_inventory or not sources:
            return []
        exclude_ids = exclude_ids or set()
        pool = [
            item
            for item in self._chunk_inventory
            if item["id"] not in exclude_ids and item["metadata"].get("source") in sources
        ]
        if not pool:
            return []
        rng = rng or random.Random()
        rng.shuffle(pool)
        selected = pool[:n]
        return [(item["text"], item["metadata"], item["id"]) for item in selected]

def _build_embedding_function(model_name: str):
    device = _default_device() or "cpu"
    last_error: Optional[Exception] = None

    for trust_flag in (True, False):
        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "normalize_embeddings": True,
            "local_files_only": False,
        }
        if trust_flag:
            kwargs["trust_remote_code"] = True
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(**kwargs)
        except TypeError:
            last_error = TypeError("Unsupported SentenceTransformer init kwargs.")
            continue
        except Exception as err:
            last_error = err
            if not trust_flag:
                break
            continue

    fallback_model = os.getenv("FALLBACK_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LOGGER.warning(
        f"[vector-db] Failed to initialize local embedding model '{model_name}' "
        f"on device '{device}' ({type(last_error).__name__ if last_error else 'UnknownError'}: "
        f"{last_error}). Falling back to local model '{fallback_model}'."
    )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=fallback_model,
        device=device,
        normalize_embeddings=True,
        local_files_only=True,
    )

def _choose_best_text(primary: str, secondary: str) -> str:
    primary_score = _text_quality_score(primary)
    secondary_score = _text_quality_score(secondary)
    if primary_score == secondary_score:
        return primary if len(primary) >= len(secondary) else secondary
    return primary if primary_score > secondary_score else secondary

def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for ch in text if ch.isalpha())
    return letters / max(len(text), 1)

_RUNTIME_OPTIONS: Dict[str, Any] = {}
_CACHED_STORE: Optional[KnowledgeVectorStore] = None
_CACHED_FINGERPRINT: Optional[str] = None

def configure_retrieval(cfg: Dict[str, Any], project_root: Path) -> None:
    """Configure retrieval.
    
    Args:
        cfg (Dict[str, Any]): Configuration mapping that controls runtime behavior.
        project_root (Path): Resolved project directory context.
    
    Returns:
        None: No value is returned.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.tools.retrieval import configure_retrieval
        >>> configure_retrieval(...)
    
    """
    global _RUNTIME_OPTIONS

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    models_cfg = cfg.get("models", {}) or {}

    knowledge_dir = project_root / "knowledge"

    persist_dir_raw = retrieval_cfg.get("persist_dir")
    persist_dir = Path(str(persist_dir_raw)) if persist_dir_raw else None
    if persist_dir and not persist_dir.is_absolute():
        persist_dir = project_root / persist_dir

    model_name = str(
        retrieval_cfg.get("embedding_model")
        or models_cfg.get("embedding_model")
        or os.getenv("EMBEDDING_MODEL_NAME")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )

    _RUNTIME_OPTIONS = {
        "knowledge_dir": knowledge_dir,
        "collection_name": "dlgforge_knowledge",
        "chunk_size": int(retrieval_cfg.get("chunk_size", 750) or 750),
        "overlap": int(retrieval_cfg.get("overlap", 150) or 150),
        "embedding_model_name": model_name,
        "persist_dir": persist_dir,
        "rebuild_index": _as_bool(retrieval_cfg.get("rebuild_index", False), default=False),
        "skip_if_unchanged": _as_bool(retrieval_cfg.get("skip_if_unchanged", True), default=True),
        "default_k": int(retrieval_cfg.get("default_k", 4) or 4),
        "reranker_model": str(models_cfg.get("reranker_model") or ""),
        "reranker_candidates": int(models_cfg.get("reranker_candidates", 12) or 12),
        "reranker_batch_size": int(models_cfg.get("reranker_batch_size", 16) or 16),
    }

def _options_fingerprint() -> str:
    payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in _RUNTIME_OPTIONS.items()
        if key not in {"default_k", "reranker_model", "reranker_candidates", "reranker_batch_size"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def get_vector_store() -> KnowledgeVectorStore:
    """Get vector store.
    
    
    Returns:
        KnowledgeVectorStore: Value produced by this API.
    
    Raises:
        RuntimeError: Raised when validation or runtime requirements are not met.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.tools.retrieval import get_vector_store
        >>> get_vector_store(...)
    
    """
    global _CACHED_STORE, _CACHED_FINGERPRINT

    if not _RUNTIME_OPTIONS:
        raise RuntimeError("Retrieval is not configured. Call configure_retrieval first.")

    fingerprint = _options_fingerprint()
    if _CACHED_STORE is not None and _CACHED_FINGERPRINT == fingerprint:
        return _CACHED_STORE

    _CACHED_STORE = KnowledgeVectorStore(
        knowledge_dir=_RUNTIME_OPTIONS["knowledge_dir"],
        collection_name=_RUNTIME_OPTIONS["collection_name"],
        chunk_size=_RUNTIME_OPTIONS["chunk_size"],
        overlap=_RUNTIME_OPTIONS["overlap"],
        embedding_model_name=_RUNTIME_OPTIONS["embedding_model_name"],
        persist_dir=_RUNTIME_OPTIONS["persist_dir"],
        rebuild_index=_RUNTIME_OPTIONS["rebuild_index"],
        skip_if_unchanged=_RUNTIME_OPTIONS["skip_if_unchanged"],
    )
    _CACHED_FINGERPRINT = fingerprint
    return _CACHED_STORE

def vector_db_search(query: str, k: Optional[int] = None, use_reranker: bool = False) -> Dict[str, Any]:
    """Vector db search.
    
    Args:
        query (str): Input text.
        k (Optional[int]): Numeric control value for processing behavior.
        use_reranker (bool): bool value used by this operation.
    
    Returns:
        Dict[str, Any]: Value produced by this API.
    
    Raises:
        Exception: Propagates unexpected runtime errors from downstream calls.
    
    Side Effects / I/O:
        - May read from or write to local filesystem artifacts.
    
    Preconditions / Invariants:
        - Callers should provide arguments matching annotated types and expected data contracts.
    
    Examples:
        >>> from dlgforge.tools.retrieval import vector_db_search
        >>> vector_db_search(...)
    
    """
    search_start = time.perf_counter()
    store = get_vector_store()
    default_k = int(_RUNTIME_OPTIONS.get("default_k", 4) or 4)
    k_final = k if isinstance(k, int) and k > 0 else default_k

    if use_reranker:
        passages = _rerank_passages(store, query=query, k_final=k_final)
    else:
        passages = store.similarity_search(query, k_final)

    records: List[Dict[str, Any]] = []
    for passage, metadata in passages:
        records.append(
            {
                "source": metadata.get("source", "unknown source"),
                "chunk_index": metadata.get("chunk_index"),
                "content": passage.strip(),
            }
        )

    rendered = "\n".join(f"[{row['source']}] {row['content']}" for row in records) if records else ""
    LOGGER.info(f"[retrieval] vector_db_search returned {len(records)} results in {time.perf_counter() - search_start:.2f}s")
    return {
        "query": query,
        "results": records,
        "rendered": rendered or "Knowledge base is empty or no relevant passages were found.",
    }

def _rerank_passages(store: KnowledgeVectorStore, query: str, k_final: int) -> List[Tuple[str, Dict[str, Any]]]:
    rerank_model = str(_RUNTIME_OPTIONS.get("reranker_model") or "")
    if not rerank_model:
        return store.similarity_search(query, k_final)

    candidates = max(k_final, int(_RUNTIME_OPTIONS.get("reranker_candidates", 12) or 12))
    results = store.similarity_search_with_ids(query, candidates)
    if not results:
        return []

    try:
        from sentence_transformers import CrossEncoder

        device = _default_device() or "cpu"
        model = CrossEncoder(rerank_model, device=device)
        pairs = [(query, passage) for passage, _, _ in results]
        scores = model.predict(pairs, batch_size=int(_RUNTIME_OPTIONS.get("reranker_batch_size", 16) or 16))
        ranked = sorted(zip(results, scores), key=lambda item: float(item[1]), reverse=True)
        top = [item[0] for item in ranked[:k_final]]
        return [(passage, metadata) for passage, metadata, _ in top]
    except Exception as err:
        LOGGER.warning(f"[reranker] Failed to rerank with {rerank_model}: {err}")
        return [(passage, metadata) for passage, metadata, _ in results[:k_final]]
