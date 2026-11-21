"""
rerank.py - Lightweight Cross-Encoder Reranker using FlashRank (ONNX TinyBERT / MiniLM).
======================================================================================
Computes full query-passage cross-attention scores for candidate passages
retrieved by bi-encoder first-pass search, boosting precision and MRR.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orma.store.records import Sentence

logger = logging.getLogger("orma.rerank")


class Reranker:
    """Cross-encoder reranker wrapping FlashRank ONNX models with graceful fallback."""

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path(tempfile.gettempdir()) / "flashrank_cache")
        self._ranker = None
        self._init_failed = False

    def _load(self) -> None:
        if not self.enabled or self._init_failed or self._ranker is not None:
            return
        try:
            from flashrank import Ranker

            self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("FlashRank unavailable (%s). Falling back to bi-encoder.", e)
            self._init_failed = True

    def rerank(
        self,
        query: str,
        passages: list[tuple[Sentence, float]],
        top_k: int | None = None,
    ) -> list[tuple[Sentence, float]]:
        """Reranks candidate (sentence, score) tuples using cross-encoder cross-attention.
        If reranker is disabled or unavailable, returns passages unchanged.
        """
        if not self.enabled or not passages or not query.strip():
            return passages[:top_k] if top_k else passages

        self._load()
        if self._ranker is None:
            return passages[:top_k] if top_k else passages

        try:
            from flashrank import RerankRequest

            payload = [{"id": i, "text": s.text} for i, (s, _score) in enumerate(passages)]
            req = RerankRequest(query=query, passages=payload)
            results = self._ranker.rerank(req)

            # Build reranked list
            reranked: list[tuple[Sentence, float]] = []
            for r in results:
                idx = int(r["id"])
                sentence, orig_score = passages[idx]
                cross_score = float(r["score"])
                fused_score = cross_score * (orig_score if orig_score > 0 else 1.0)
                reranked.append((sentence, fused_score))

            reranked.sort(key=lambda p: -p[1])
            return reranked[:top_k] if top_k else reranked
        except Exception as e:  # noqa: BLE001
            logger.warning("Reranking failed (%s). Using original ranking.", e)
            return passages[:top_k] if top_k else passages
