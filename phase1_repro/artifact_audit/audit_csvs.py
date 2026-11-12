"""Audit every provided CSV: shape, dtypes, id columns, and resolution of chunk IDs against doc/table chunks."""
import ast, json, re, sys, collections
from pathlib import Path
import pandas as pd

raw = Path(sys.argv[1]); scratch = Path(sys.argv[2])
doc_ids = set(json.load(open(scratch / "chunk_ids.json")))
tab_ids = set(json.load(open(scratch / "table_ids.json")))
ORD = re.compile(r"^\d+(st|nd|rd|th)$")


def lit(x):
    if isinstance(x, str) and x.startswith("["):
        try: return ast.literal_eval(x)
        except Exception: return None
    return None


def classify(i):
    if i in doc_ids: return "doc"
    if i in tab_ids: return "table"
    if isinstance(i, str) and i.startswith("ottaqa-intro-"): return "intro(unresolved)"
    return "unresolved"


summary = {}
qsets = {}
for f in sorted(raw.glob("*.csv")):
    df = pd.read_csv(f)
    print(f"\n==================== {f.name}  rows={len(df)} cols={df.shape[1]}")
    print("dtypes:", dict(collections.Counter(str(t) for t in df.dtypes)))
    nonord = [c for c in df.columns if not ORD.match(c)]
    ords = [c for c in df.columns if ORD.match(c)]
    print("non-ordinal columns + dtype:", {c: str(df[c].dtype) for c in nonord})
    print("ordinal rank columns:", f"{ords[0]}..{ords[-1]} ({len(ords)})" if ords else None)
    print("question_id unique:", df["question_id"].nunique(), "| null question_id:", df["question_id"].isna().sum())
    qsets[f.name] = set(df["question_id"])
    # rank-column ID resolution
    if ords:
        vals = df[ords].values.ravel()
        nn = [v for v in vals if isinstance(v, str)]
        print("rank cells non-null:", len(nn), "/", len(vals), "| nulls:", len(vals) - len(nn))
        c = collections.Counter(classify(v) for v in nn)
        print("rank cell resolution:", dict(c))
        # looks like score?
        num = sum(1 for v in nn if re.fullmatch(r"-?\d+(\.\d+)?", v))
        print("rank cells that are numeric:", num)
        # duplicates within row
        dup_rows = sum(1 for r in df[ords].values if len([x for x in r if isinstance(x, str)]) != len({x for x in r if isinstance(x, str)}))
        print("rows with duplicate ids in rank columns:", dup_rows)
        filled = df[ords].notna().sum(axis=1)
        print("filled rank cols per row min/median/max:", filled.min(), int(filled.median()), filled.max())
        ex_unres = [v for v in nn if classify(v) != "doc" and classify(v) != "table"][:3]
        print("unresolved examples:", ex_unres)
    # gold
    if "gold_docs" in df:
        g = df["gold_docs"].map(lit)
        bad = g.isna().sum()
        flat = [x for l in g.dropna() for x in l]
        print("gold_docs parse failures:", bad, "| gold ids total:", len(flat), "| per-q min/median/max:",
              g.dropna().map(len).min(), int(g.dropna().map(len).median()), g.dropna().map(len).max())
        print("gold resolution:", dict(collections.Counter(classify(x) for x in flat)))
        print("gold unresolved examples:", [x for x in flat if classify(x) in ("unresolved", "intro(unresolved)")][:3])
    # list columns
    for c in nonord:
        if c.endswith("_results") or c.endswith("_chunk_ids") or c.endswith("communities"):
            s = df[c].dropna()
            parsed = s.map(lit)
            if parsed.notna().any():
                lens = parsed.dropna().map(len)
                flat = [x for l in parsed.dropna() for x in l]
                res = dict(collections.Counter(classify(x) for x in flat)) if c.endswith("_chunk_ids") else None
                print(f"  list col {c}: len min/med/max {lens.min()}/{int(lens.median())}/{lens.max()} sample={flat[:3]} resolution={res}")
            else:
                print(f"  str col {c}: sample={s.iloc[0]!r:.120}")
    # consistency: topN_chunk_ids == first N ordinal cols?
    for n in (2, 3, 5, 100):
        c = f"top{n}_chunk_ids"
        if c in df and ords and len(ords) >= n:
            eq = sum(1 for _, r in df.iterrows() if lit(r[c]) == [x for x in r[ords[:n]].tolist() if isinstance(x, str)])
            print(f"  {c} == first {n} rank cols: {eq}/{len(df)}")
    # numeric columns summary
    num = df.select_dtypes("number")
    if len(num.columns):
        print("numeric cols summary:\n", num.describe().T[["mean", "min", "max"]].round(4).to_string()[:4000])
    summary[f.name] = len(df)

print("\nquestion_id set comparisons:")
names = list(qsets)
for a in names:
    print(a, {b: len(qsets[a] & qsets[b]) for b in names})
