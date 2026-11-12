"""Cross-compatibility deep dive: why IDs are unresolved, gold differences, seed vs community relationships."""
import ast, json, os, re, sys, collections
from pathlib import Path
import pandas as pd

raw = Path(sys.argv[1]); docs_dir = Path(sys.argv[2]); scratch = Path(sys.argv[3])
doc_ids = set(json.load(open(scratch / "chunk_ids.json")))
tables = json.load(open(raw / "table_chunks_with_metadata.json"))
tab_ids = {t["chunk_id"] for t in tables}
tab_src = {t["source_id"] for t in tables}
allids = doc_ids | tab_ids
L = lambda s: ast.literal_eval(s) if isinstance(s, str) else []
ORD = lambda n: [f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th" for i in range(1, n + 1)]
strip = lambda i: re.sub(r"_[0-9a-f]{12}$", "", i)

# prefix (id without trailing 12-hex hash) index
pref = collections.defaultdict(set)
for i in allids: pref[strip(i)].add(i)

# structural links in docs
sl_targets = collections.Counter(); sl_in_tab = 0; n = 0
for f in os.listdir(docs_dir):
    o = json.load(open(docs_dir / f)); n += 1
    for t in o["source_info"]["structural_link"]:
        sl_targets[t] += 1; sl_in_tab += t in tab_src
print(f"structural_link targets: {len(sl_targets)} distinct; links whose target is a table source_id in table file: {sl_in_tab}/{sum(sl_targets.values())}")
print("table chunks:", len(tab_ids), "distinct table source_ids:", len(tab_src))
print("table chunk_index distribution:", collections.Counter(t["chunk_index"] for t in tables).most_common(5))

avg = pd.read_csv(raw / "dense_sparse_average_results.csv")
bm = pd.read_csv(raw / "bm25_only_results.csv")
seq = pd.read_csv(raw / "dense_sparse_sequential_results.csv")
cexp = pd.read_csv(raw / "community_expanded_dense_sparse_results.csv")
ck = pd.read_csv(raw / "community_k_plus_k1_dense_sparse_results.csv")
enh = pd.read_csv(raw / "enhanced_results_final_communities_2049.csv")

def why(i):
    if i in allids: return "resolved"
    if i.startswith("ottaqa-intro-"): return "intro chunk (no file)"
    if strip(i) in pref: return "same prefix, different hash"
    return "absent entirely"

for name, df, k in [("avg", avg, 100), ("bm25", bm, 10), ("seq", seq, 10)]:
    cells = [x for x in df[ORD(k)].values.ravel() if isinstance(x, str)]
    print(f"\n{name} rank-cell reasons:", dict(collections.Counter(why(x) for x in cells)))
    gold = [x for g in df.gold_docs for x in L(g)]
    print(f"{name} gold reasons:", dict(collections.Counter(why(x) for x in gold)))
    ex = [x for x in cells if why(x) == "absent entirely"][:4]
    print(f"{name} absent examples:", ex)

# rows/questions whose gold is fully resolvable
for name, df in [("avg", avg), ("bm25", bm)]:
    full = sum(all(x in allids for x in L(g)) for g in df.gold_docs)
    anyr = sum(any(x in allids for x in L(g)) for g in df.gold_docs)
    docs_only = sum(all(x in doc_ids for x in L(g)) for g in df.gold_docs)
    print(f"{name}: questions with ALL gold resolvable {full}, ANY {anyr}, all gold are doc chunks {docs_only} / {len(df)}")

# gold differences between files
m = avg.set_index("question_id").gold_docs.map(L)
for name, df in [("bm25", bm), ("seq", seq), ("cexp", cexp), ("ck", ck)]:
    o = df.set_index("question_id").gold_docs.map(L)
    same = sum(m[q] == o[q] for q in m.index)
    print(f"gold_docs identical avg vs {name}: {same}/{len(m)}")
diffq = [q for q in m.index if m[q] != bm.set_index("question_id").gold_docs.map(L)[q]][:2]
for q in diffq:
    print("DIFF", q, "\n avg :", m[q], "\n bm25:", L(bm.set_index('question_id').gold_docs[q]))

# question text identical?
print("question text identical avg vs enh:", (avg.set_index("question_id").question == enh.set_index("question_id").question.reindex(avg.question_id)).sum())

# seq/bm25 top10 vs avg top10
a10 = avg.set_index("question_id")[ORD(10)]
s10 = seq.set_index("question_id")[ORD(10)]
print("seq top10 == avg top10 (as list):", sum(list(a10.loc[q]) == list(s10.loc[q]) for q in a10.index),
      "| same set:", sum(set(a10.loc[q]) == set(s10.loc[q]) for q in a10.index))

# community files vs avg
A = avg.set_index("question_id"); C = cexp.set_index("question_id"); K = ck.set_index("question_id")
print("cexp rank cols == ck rank cols (all 100, NaN-aware):",
      sum(C.loc[q, ORD(100)].fillna("").tolist() == K.loc[q, ORD(100)].fillna("").tolist() for q in A.index))
for kk in (1, 2, 5, 10, 20):
    print(f"cexp first {kk} == avg first {kk}: {sum(C.loc[q, ORD(kk)].tolist() == A.loc[q, ORD(kk)].tolist() for q in A.index)}")
# overlap of community list with avg seeds; fraction of cexp ids that appear in avg top100
inter = []; newfrac = []
for q in A.index:
    c = [x for x in C.loc[q, ORD(100)] if isinstance(x, str)]
    a = set(A.loc[q, ORD(100)])
    inter.append(len(set(c) & a) / max(1, len(c)))
print("mean fraction of cexp ids also in avg top100:", round(sum(inter) / len(inter), 3))
filled = C[ORD(100)].notna().sum(axis=1)
print("cexp filled-count distribution:", filled.value_counts().sort_index().head(8).to_dict(), "... rows with 100:", int((filled == 100).sum()))
# top100_chunk_ids mismatch cause
q = next(q for q in C.index if C.loc[q, ORD(100)].isna().any())
t100 = L(C.loc[q, "top100_chunk_ids"]) if C.loc[q, "top100_chunk_ids"].startswith("[") else None
print("example row with NaNs: filled", int(C.loc[q, ORD(100)].notna().sum()), "| top100_chunk_ids sample tail:", (t100 or [])[-3:],
      "| unique in top100 list:", len(set(t100 or [])))
# results (table ids) columns: relation to chunk ids
r = L(A.iloc[0]["top5_results"]); c = L(A.iloc[0]["top5_chunk_ids"])
print("avg top5_results vs top5_chunk_ids row0:", list(zip(r, c)))
# table_id column: is it the table part of gold?
tid_in_gold = sum(any(strip(x).replace("ottaqa-", "").startswith(t.replace("(", "_").replace(")", "_").replace(".", "_")) for x in L(g))
                  for t, g in zip(avg.table_id, avg.gold_docs))
print("table_id appears as prefix of some gold id:", tid_in_gold, "/", len(avg))
# community metrics in enh: sanity — which gold def
print("enh distinct communities referenced (top10):",
      len({c for s in enh.top_10_communities.dropna() for c in s.split("|")}))
