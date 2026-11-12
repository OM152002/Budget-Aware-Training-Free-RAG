"""Top-m candidate pre-filter = SAGE's own neighbor ranker, reused unmodified.

Scoring: scripts/analysis/graph_enhanced_accuracy_analyzer.py
  GraphEnhancedAccuracyAnalyzer._calculate_hybrid_similarities (L515-562)
    = 0.40 * [cos(q, content) + cos(q, topic) + entity/event bonuses]   (_calculate_document_similarities, L659-700)
    + 0.60 * BM25 fit on the candidate set, max-normalized             (_calculate_bm25_similarities + BM25, L564-587, L83-124)

Approved decisions:
  - question entity/event lists are empty (A6 question metadata unavailable) -> bonuses are 0;
  - document candidates only (the adapter's corpus is documents only);
  - candidates are passed in the adapter's deterministic order; final order is a stable sort on (-score, chunk_id);
  - SAGE's in-memory embedding cache (GPUEmbeddingService.embedding_cache) may be used only after verification
    that it gives rankings identical to cache-free scoring (see verify_cache.py).

The analyzer is constructed with a cache_dir outside sage/; nothing is saved to disk, and no SAGE file is modified.
chunk_metadata is filled with exactly the document fields _load_chunk_metadata (L389-395) builds. Difference: the
adapter stores a missing/None `events` as {} (3 chunks have events=None; SAGE's own loader would keep None and
crash in list(None.keys()) at L689). With empty question events, both give a 0 bonus.
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "sage"))
sys.path.insert(0, str(PROJECT / "traversal_adapter"))
# Note: SAGE calls encode(show_progress_bar=True), so progress bars go to stderr; redirect stderr when running.

import logging  # noqa: E402

from scripts.analysis.graph_enhanced_accuracy_analyzer import GraphEnhancedAccuracyAnalyzer  # noqa: E402

logging.getLogger("scripts.analysis.graph_enhanced_accuracy_analyzer").setLevel(logging.WARNING)


class SagePrefilter:
    def __init__(self, adapter, use_cache: bool = True, cache_dir: Optional[Path] = None):
        cache_dir = Path(cache_dir or Path(__file__).resolve().parent / "analyzer_cache_dir")
        # Only __init__ runs (paths are stored, model loaded on CPU); load_all_data() is never called.
        self.analyzer = GraphEnhancedAccuracyAnalyzer(cache_dir=str(cache_dir))
        self.use_cache = use_cache
        self.analyzer.chunk_metadata = {
            cid: {"type": "document", "content": c[0], "topic": c[3], "entities": c[4], "events": c[5]}
            for cid, c in adapter.chunks.items()
        }

    def clear_cache(self):
        self.analyzer.embedding_service.embedding_cache.clear()

    def score(self, question: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """All candidates, ranked by SAGE hybrid score (desc), ties broken by chunk_id (asc)."""
        if not candidate_ids:
            return []
        if not self.use_cache:
            self.clear_cache()
        scored = self.analyzer._calculate_hybrid_similarities(question, [], [], list(candidate_ids))
        return sorted(scored, key=lambda x: (-x[1], x[0]))

    def top_m(self, question: str, candidate_ids: List[str], m: int) -> List[Tuple[str, float]]:
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        return self.score(question, candidate_ids)[:m]


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t
