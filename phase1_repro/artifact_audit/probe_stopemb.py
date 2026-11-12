"""Test: stored embedding == MiniLM(normalize, stopword-removed content) using SAGE's own function."""
import json, os, sys, random, numpy as np
sage, d = sys.argv[1], sys.argv[2]; sys.path.insert(0, sage)
from loguru import logger; logger.remove()
import src.pipeline  # noqa
from src.pipeline.data.stopwords_loader import ENGLISH_STOPWORDS
import re
from sentence_transformers import SentenceTransformer
def strip(_self, text):  # verbatim logic of parallel_document_processor.py L979-1004 (module needs `ray`)
    if not text.strip(): return text
    words = re.findall(r'\b\w+\b', text.lower())
    cleaned = ' '.join(w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 2)
    return cleaned if cleaned.strip() else text
fs = sorted(os.listdir(d)); random.seed(0); pick = random.sample(fs, 300)
objs = [json.load(open(os.path.join(d, f))) for f in pick]
E = np.array([o["embedding"] for o in objs], dtype=np.float32)
texts = [strip(None, o["content"]) for o in objs]
for dev in ("cpu", "mps"):
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    X = m.encode(texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True)
    cos = (X * E).sum(1); dmax = np.abs(X - E).max(1)
    print(f"[{dev}] stopword-stripped: cos min {cos.min():.6f} p1 {np.quantile(cos,.01):.6f} median {np.median(cos):.6f} | max|Δ| median {np.median(dmax):.2e} max {dmax.max():.2e} | n(cos<0.999)={int((cos<0.999).sum())}")
    if dev == "mps":
        import torch
        m2 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps"); m2.half()
        X2 = m2.encode(texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True).astype(np.float32)
        c2 = (X2 * E).sum(1); print(f"[mps fp16] cos min {c2.min():.6f} median {np.median(c2):.6f} max|Δ| max {np.abs(X2-E).max():.2e}")
bad = [(i, float(c)) for i, c in enumerate((X * E).sum(1)) if c < 0.999][:3]
for i, c in bad: print("LOW", c, objs[i]["chunk_id"][:80], "| merged_sentence_count", objs[i]["merged_sentence_count"], "| len", len(objs[i]["content"]))
