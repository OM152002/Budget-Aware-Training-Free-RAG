"""Evaluate regenerated seeds: determinism (run A vs B), Recall@k / Hit@k over document gold, and a
diagnostic-only overlap with the compatible portion of the authors' dense_sparse_average ranking (S6; no tuning).

Usage: python -I evaluate_seeds.py <runA_dir> <runB_dir>
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_common as C  # noqa: E402

KS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
ORD = lambda n: [f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th" for i in range(1, n + 1)]


def ranked(df):
    return [[x for x in r if isinstance(x, str)] for r in df[ORD(100)].values]


def metrics(rank_lists, gold_lists, mask):
    out = {}
    idx = [i for i, m in enumerate(mask) if m]
    for k in KS:
        rec = [len(set(rank_lists[i][:k]) & set(gold_lists[i])) / len(gold_lists[i]) for i in idx]
        hit = [bool(set(rank_lists[i][:k]) & set(gold_lists[i])) for i in idx]
        out[f"@{k}"] = {"recall": round(float(np.mean(rec)), 4), "hit": round(float(np.mean(hit)), 4)}
    return {"n_questions": len(idx), **out}


def main():
    a_dir, b_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    A, B = pd.read_csv(a_dir / "seeds_top100.csv"), pd.read_csv(b_dir / "seeds_top100.csv")
    sa = (a_dir / "seeds_top100_scores.jsonl").read_text().splitlines()
    sb = (b_dir / "seeds_top100_scores.jsonl").read_text().splitlines()
    rep = {"determinism": {"seed_csv_identical": A.equals(B), "score_jsonl_identical": sa == sb,
                           "questions": len(A), "all_returned_100": bool((A.n_returned == 100).all())}}

    rl = ranked(A)
    doc_gold = [json.loads(x) for x in A.doc_gold]
    all_gold = [ast.literal_eval(x) for x in A.gold_docs]
    n_doc = A.n_doc_gold.to_numpy(); n_all = A.n_gold.to_numpy()
    rep["eval_sets"] = {
        "PRIMARY doc-retrieval: questions with >=1 doc gold; denominator = doc gold only":
            metrics(rl, doc_gold, n_doc > 0),
        "SECONDARY all-gold-are-docs subset": metrics(rl, doc_gold, (n_doc > 0) & (n_doc == n_all)),
        "TRANSPARENCY all 2,212 questions; denominator = all gold incl. tables/unresolved (unreachable -> misses)":
            metrics(rl, all_gold, np.ones(len(A), bool)),
    }
    rep["question_counts"] = {"total": len(A), "with_doc_gold": int((n_doc > 0).sum()),
                              "table_only_or_unresolved_excluded_from_primary": int((n_doc == 0).sum())}

    # S6 diagnostic: authors' ranking restricted to IDs that resolve to our corpus, order preserved
    corpus = set(pd.read_csv(C.GRAPH_IDS_CSV).chunk_id)
    auth = pd.read_csv(C.AUTHOR_RANKING_CSV).set_index("question_id").loc[A.question_id]
    arl = ranked(auth)
    diag = {}
    for k in [5, 10, 20, 50, 100]:
        resolvable_frac, overlap, cover = [], [], []
        for ours, theirs in zip(rl, arl):
            t_k = theirs[:k]
            t_res = [x for x in t_k if x in corpus]
            resolvable_frac.append(len(t_res) / len(t_k))
            if t_res:
                overlap.append(len(set(t_res) & set(ours[:k])) / len(t_res))
            comp = [x for x in theirs if x in corpus][:k]
            if comp:
                cover.append(len(set(comp) & set(ours[:k])) / len(comp))
        diag[f"@{k}"] = {"authors_topk_resolvable_frac": round(float(np.mean(resolvable_frac)), 4),
                         "share_of_authors_resolvable_topk_in_our_topk": round(float(np.mean(overlap)), 4),
                         "share_of_authors_first_k_resolvable_in_our_topk": round(float(np.mean(cover)), 4)}
    exact_top1 = float(np.mean([bool(t) and bool(o) and (lambda c: c[0] == o[0] if c else False)([x for x in t if x in corpus])
                                for o, t in zip(rl, arl)]))
    diag["top1_equal_to_authors_first_resolvable"] = round(exact_top1, 4)
    # context only: authors' raw ranking scored against the same primary doc gold
    diag["CONTEXT_ONLY_authors_ranking_primary_metric"] = metrics(arl, doc_gold, n_doc > 0)
    rep["S6_diagnostic_overlap_with_authors_ranking"] = diag
    (a_dir / "evaluation.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
