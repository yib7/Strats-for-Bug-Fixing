"""Exemplar retrievers for the RAG pipeline: BM25 and CodeBERT (dense).

Both classes share the interface ``index(pairs) -> None`` / ``retrieve(query,
k) -> list[dict]`` where ``pairs`` and the returned exemplars are
``{"buggy": str, "fixed": str}`` dicts.

**Leakage guard**: the knowledge base indexed here must be the ``train``
split only. These classes do not enforce that at runtime (they take whatever
pairs they're given) -- the guard lives in the ``pop rag`` CLI path, which
refuses to index a non-``train`` split unless explicitly overridden.
"""

from __future__ import annotations

import re
from typing import Protocol

import bm25s
import faiss
import numpy as np

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize_code(text: str) -> list[str]:
    """Simple code-aware tokenization: split on identifiers/punctuation, lowercase.

    Punctuation and whitespace act as separators; alphanumeric/underscore runs
    (identifiers, keywords, numbers) become tokens.
    """
    return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]


class EncodeFn(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


class BM25Retriever:
    """Sparse lexical retriever over the ``buggy`` side of KB pairs, via bm25s."""

    def __init__(self) -> None:
        self._model: bm25s.BM25 | None = None
        self._pairs: list[dict] = []

    def index(self, pairs: list[dict]) -> None:
        self._pairs = list(pairs)
        corpus_tokens = [tokenize_code(pair["buggy"]) for pair in self._pairs]
        model = bm25s.BM25()
        model.index(corpus_tokens, show_progress=False)
        self._model = model

    def retrieve(self, query: str, k: int) -> list[dict]:
        if self._model is None:
            raise RuntimeError("BM25Retriever.retrieve() called before index()")
        if k <= 0 or not self._pairs:
            return []

        k_eff = min(k, len(self._pairs))
        query_tokens = [tokenize_code(query)]
        results, _scores = self._model.retrieve(query_tokens, k=k_eff, show_progress=False)
        return [self._pairs[i] for i in results[0]]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class CodeBERTRetriever:
    """Dense retriever: microsoft/codebert-base mean-pooled embeddings + FAISS IP.

    Model loading is lazy and injectable: pass ``encode_fn`` (a callable
    ``list[str] -> np.ndarray``) to avoid ever importing torch/transformers or
    downloading the model, which is what tests do. Without an injected
    ``encode_fn``, the default encoder lazily loads ``model_name`` on first use
    and runs on the GPU when one is available.

    ``index`` encodes the knowledge base in ``batch_size`` chunks rather than one
    call: the CodeXGLUE train split is ~52k pairs, and encoding them all at once
    builds a single ``[N, seq, hidden]`` tensor tens of GB large -- on the CPU
    that gets the process OOM-killed (SIGKILL / exit -9). Chunking bounds peak
    memory to one batch; ``encode_fn`` is therefore called once per chunk.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        encode_fn: EncodeFn | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._encode_fn = encode_fn
        self._batch_size = batch_size
        self._pairs: list[dict] = []
        self._index: faiss.IndexFlatIP | None = None

    def _default_encode_fn(self) -> EncodeFn:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModel.from_pretrained(self._model_name)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        def encode(texts: list[str]) -> np.ndarray:
            with torch.no_grad():
                batch = tokenizer(
                    texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(device)
                outputs = model(**batch)
                mask = batch["attention_mask"].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean_pooled = summed / counts
            return mean_pooled.cpu().numpy()

        return encode

    def _get_encode_fn(self) -> EncodeFn:
        if self._encode_fn is None:
            self._encode_fn = self._default_encode_fn()
        return self._encode_fn

    def index(self, pairs: list[dict]) -> None:
        self._pairs = list(pairs)
        if not self._pairs:
            self._index = None
            return

        encode = self._get_encode_fn()
        texts = [pair["buggy"] for pair in self._pairs]
        # Encode in batches so a large KB never builds one giant tensor (see class docstring).
        chunks = [
            np.asarray(encode(texts[i : i + self._batch_size]), dtype="float32")
            for i in range(0, len(texts), self._batch_size)
        ]
        vecs = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        vecs = _l2_normalize(vecs)

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._index = index

    def retrieve(self, query: str, k: int) -> list[dict]:
        if self._index is None:
            raise RuntimeError("CodeBERTRetriever.retrieve() called before index()")
        if k <= 0 or not self._pairs:
            return []

        encode = self._get_encode_fn()
        qvec = np.asarray(encode([query]), dtype="float32")
        qvec = _l2_normalize(qvec)

        k_eff = min(k, len(self._pairs))
        _scores, idxs = self._index.search(qvec, k_eff)
        return [self._pairs[i] for i in idxs[0] if i != -1]
