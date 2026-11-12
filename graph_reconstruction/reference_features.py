"""Independent reimplementation of SAGE's doc-doc pair features, written from the released code's semantics
(graph_analysis_pipeline.py L638-899, L921-958, L490-556) without calling any SAGE method.

Used only for validation. Float32 rounding of the fuzzy score mirrors SAGE's torch.float32 similarity
tensor (L685) so threshold comparisons at 0.75 behave identically; ties resolve to the first maximum.
"""
import re

import numpy as np


def normalize_name(name):
    s = re.sub(r"\s+", " ", name.lower().strip())
    return re.sub(r"[^\w\s]", "", s)


def fuzzy(s1, s2):
    c1, c2 = set(s1.replace(" ", "")), set(s2.replace(" ", ""))
    if not c1 or not c2:
        return 0.0
    jac = len(c1 & c2) / len(c1 | c2)
    lens = min(len(s1), len(s2)) / max(len(s1), len(s2)) if max(len(s1), len(s2)) else 0.0
    return float(np.float32(0.7 * jac + 0.3 * lens))


def greedy_match(names1, names2, threshold=0.75):
    n1 = [normalize_name(x) for x in names1]
    n2 = [normalize_name(x) for x in names2]
    used, matches = set(), []
    for i, a in enumerate(n1):
        best_j, best = None, None
        for j, b in enumerate(n2):
            if j in used:
                continue
            s = fuzzy(a, b)
            if best is None or s > best:
                best, best_j = s, j
        if best_j is not None and best >= threshold:
            matches.append((i, best_j))
            used.add(best_j)
    return matches


def _hashable(x):
    if isinstance(x, list):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    if isinstance(x, set):
        return tuple(sorted(_hashable(v) for v in x))
    return x


def _rel_set(item):
    r = item.get("relationships", [])
    if isinstance(r, list):
        return {_hashable(v) for v in r}
    return {_hashable(r)} if r else set()


def entity_features(ent1, ent2):
    """-> (entity_count, entity_relationship_overlap) or None when SAGE would not compute them."""
    if not ent1 or not ent2:
        return None
    names1, names2 = list(ent1), list(ent2)
    d1, d2 = list(ent1.values()), list(ent2.values())
    matches = greedy_match(names1, names2)
    overlap = {}
    for i, j in matches:
        a, b = d1[i], d2[j]
        if isinstance(a, dict) and isinstance(b, dict):
            if "relationships" in a or "relationships" in b:
                r1, r2 = _rel_set(a), _rel_set(b)
                overlap[names1[i]] = len(r1 & r2) / max(len(r1), len(r2), 1)
            else:
                overlap[names1[i]] = 0.0
    avg = sum(overlap.values()) / len(overlap) if overlap else 0.0
    return len(matches), float(avg)


def event_count(ev1, ev2):
    if not ev1 or not ev2:
        return 0
    return len(greedy_match(list(ev1), list(ev2)))


def pair_features(c1, c2, emb, topic_emb, idx):
    """c1, c2: raw doc JSON dicts (c1 is the lexicographically smaller id, as SAGE orders pairs)."""
    i, j = idx[c1["chunk_id"]], idx[c2["chunk_id"]]
    e1, e2 = emb[i].astype(np.float64), emb[j].astype(np.float64)
    e1, e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    t1, t2 = topic_emb[i].astype(np.float64), topic_emb[j].astype(np.float64)
    t1, t2 = t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2)
    m1, m2 = c1.get("metadata") or {}, c2.get("metadata") or {}
    ent = entity_features(m1.get("entities") or {}, m2.get("entities") or {})
    ec, ov = ent if ent else (0, 0.0)
    return {
        "content_similarity": max(0.0, float(e1 @ e2)),
        "topic_similarity": max(0.0, float(t1 @ t2)),
        "entity_count": ec,
        "entity_relationship_overlap": ov,
        "event_count": event_count(m1.get("events", {}), m2.get("events", {})),
    }
