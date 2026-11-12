"""Figures F1–F5 (ANALYSIS_PLAN.md / EXPERIMENT_PLAN §7), built ONLY from the canonical scores and statistics.

Inputs (hash-checked against their manifests): analysis/scores/{aggregates.json, per_question.jsonl},
analysis/stats/{comparisons.json, absolute_recall_ci.json}. No metric is recomputed; T1–T6 are not touched.
  F1  ID Recall vs budget B — (a) SUB-300, every method at B∈{1,2,4,8}; (b) TEST-1704 baselines + M5 at B=4
  F2  ID Recall vs cost, one point per (method, B), Pareto frontier — SUB-300; (a) x = expansions, (b) x = latency (D5)
  F3  Gold passages in the final evidence by hop depth (0 / 1 / 2+), M4 vs M5 — TEST-1704, B=4; (a) all, (b) by category
  F4  Pre-filter rank of the SLM's selected candidate, by step — TEST-1704, M5 B=4 (M4 is always rank 1)
  F5  Per-question paired outcome M5 vs M4 (win/tie/loss) by category — TEST-1704, B=4 maximum budget (T1 comparison)
Validation reads the plotted data back from the matplotlib artists and compares it with the canonical sources.

Outputs: analysis/figures/F1..F5.{png,pdf}, F*_data.csv, figures_validation.json, figures_manifest.json
Usage: .venv-sage/bin/python -P -s analysis/figures.py
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection, PolyCollection  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
SC, ST, OUT = PROJECT / "analysis/scores", PROJECT / "analysis/stats", PROJECT / "analysis/figures"
CATS = ("all_gold_in_seeds", "reachable_top20", "reachable_filtered", "unreachable_1hop")
CAT_LABEL = {"all_gold_in_seeds": "All gold\nin seeds", "reachable_top20": "Gold reachable\nin top-20",
             "reachable_filtered": "Reachable but\nfiltered out", "unreachable_1hop": "Not reachable\nin 1 hop"}
# reference palette (dataviz skill), fixed method -> hue mapping; validated: all hard gates pass (contrast WARN -> legend + T3 table view)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STYLE = {"M5": ("#2a78d6", "o", "M5 SLM (frozen, full)"), "M4": ("#eb6834", "s", "M4 SAGE-score greedy"),
         "M1": ("#1baf7a", "^", "M1 flat top-(3+b)"), "M2": ("#eda100", "D", "M2 SAGE one-shot (faithful)"),
         "M3_mean": ("#e87ba4", "v", "M3 random (mean of 3 seeds)"), "M0": (MUTED, "P", "M0 seeds only")}
HOP = {"hop0": ("#86b6ef", INK, "Hop 0: in the seeds"), "hop1": ("#2a78d6", "#ffffff", "Hop 1: neighbor of a seed"),
       "hop2plus": ("#104281", "#ffffff", "Hop 2+: multi-hop")}
WIN, LOSS = "#2a78d6", "#e34948"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ck(m, v=None, B=4, comp="max_budget", s=None):
    return f"{m}|{v}|{B}|{comp}|{s}"


def mkey(m, B, comp="max_budget"):
    return ck("M2", "faithful", B, comp) if m == "M2" else ck(m, None, 0 if m == "M0" else B, comp)


def verify():
    for d, m in ((SC, "scoring_manifest.json"), (ST, "stats_manifest.json")):
        man = json.loads((d / m).read_text())
        bad = [k for k, v in man["outputs_sha256"].items() if sha(d / k) != v]
        if bad:
            raise SystemExit(f"{d.name} changed since produced: {bad}")


def load():
    AG = {}
    for a in json.loads((SC / "aggregates.json").read_text()):
        AG[(ck(a["method"], a["variant"], a["B"], a["comparison"], a["random_seed"]), a["population"])] = a
    ABS = {(a["condition"], a["population"]): a for a in json.loads((ST / "absolute_recall_ci.json").read_text())}
    CMP = {(c["comparator"], c["reference"], c["population"]): c for c in json.loads((ST / "comparisons.json").read_text())["comparisons"]}
    PQ = [json.loads(l) for l in open(SC / "per_question.jsonl")]
    return AG, ABS, CMP, PQ


def footer(fig, text):
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7, color=INK2, wrap=True)


DATA = defaultdict(list)


def note(fig_id, **kw):
    DATA[fig_id].append(kw)


# ------------------------------------------------------------------------------------------------------------- F1
def f1(AG, ABS):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    Bs = [1, 2, 4, 8]
    off = {"M1": -0.06, "M2": -0.03, "M3_mean": 0.0, "M4": 0.03, "M5": 0.06}
    # (a) SUB-300
    ax = axes[0]
    for m in ("M1", "M2", "M3_mean", "M4", "M5"):
        col, mk, lab = STYLE[m]
        xs = [B * 2 ** off[m] for B in Bs]
        ys = [AG[(mkey(m, B), "SUB-300")]["id_recall"] for B in Bs]
        cis = [ABS[(mkey(m, B), "SUB-300")]["ci95"] for B in Bs]
        if m == "M3_mean":
            sd = [AG[(mkey(m, B), "SUB-300")]["id_recall_seed_sd"] for B in Bs]
            band = ax.fill_between(xs, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)], color=col, alpha=0.25, lw=0)
            band.set_gid(f"F1a|band|{m}")
        cont = ax.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, cis)], [c[1] - y for y, c in zip(ys, cis)]], color=col,
                           marker=mk, ms=6, lw=2 if m == "M5" else 1.5, elinewidth=0.8, capsize=2, label=lab,
                           mec=SURF, mew=0.8, zorder=4 if m == "M5" else 3)
        cont.lines[0].set_gid(f"F1a|line|{m}")
        cont.lines[2][0].set_gid(f"F1a|ci|{m}")
        for B, y, c in zip(Bs, ys, cis):
            note("F1", panel="a SUB-300", series=m, B=B, id_recall=y, ci_low=c[0], ci_high=c[1])
    m0 = AG[(mkey("M0", 0), "SUB-300")]["id_recall"]
    ax.axhline(m0, color=MUTED, ls="--", lw=1, gid="F1a|M0").set_label(STYLE["M0"][2])
    note("F1", panel="a SUB-300", series="M0", B=0, id_recall=m0)
    ax.set_title("(a) SUB-300 (n = 300): every method at B = 1, 2, 4, 8\nerror bars = 95% bootstrap CI; M3 band = ± SD over 3 seeds",
                 fontsize=9, color=INK, loc="left")
    # (b) TEST-1704
    ax = axes[1]
    for m in ("M1", "M2", "M3_mean", "M4"):
        col, mk, lab = STYLE[m]
        bs = [B for B in Bs if (mkey(m, B), "TEST-1704") in AG]
        xs = [B * 2 ** off[m] for B in bs]
        ys = [AG[(mkey(m, B), "TEST-1704")]["id_recall"] for B in bs]
        if m == "M3_mean":
            sd = [AG[(mkey(m, B), "TEST-1704")]["id_recall_seed_sd"] for B in bs]
            ax.fill_between(xs, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)], color=col, alpha=0.25, lw=0,
                            gid=f"F1b|band|{m}")
        ax.plot(xs, ys, color=col, marker=mk, ms=6, lw=1.5, label=lab, mec=SURF, mew=0.8, gid=f"F1b|line|{m}")
        for B, y in zip(bs, ys):
            note("F1", panel="b TEST-1704", series=m, B=B, id_recall=y)
    y5, c5 = AG[(mkey("M5", 4), "TEST-1704")]["id_recall"], ABS[(mkey("M5", 4), "TEST-1704")]["ci95"]
    cont = ax.errorbar([4 * 2 ** off["M5"]], [y5], yerr=[[y5 - c5[0]], [c5[1] - y5]], color=STYLE["M5"][0], marker="o", ms=8,
                       lw=0, elinewidth=1, capsize=3, label=STYLE["M5"][2] + " (B = 4 only)", mec=SURF, mew=0.8, zorder=5)
    cont.lines[0].set_gid("F1b|line|M5")
    note("F1", panel="b TEST-1704", series="M5", B=4, id_recall=y5, ci_low=c5[0], ci_high=c5[1])
    m0 = AG[(mkey("M0", 0), "TEST-1704")]["id_recall"]
    ax.axhline(m0, color=MUTED, ls="--", lw=1, gid="F1b|M0").set_label(STYLE["M0"][2])
    note("F1", panel="b TEST-1704", series="M0", B=0, id_recall=m0)
    ax.set_title("(b) TEST-1704 (n = 1,704): baselines at B = 1, 2, 4, 8; M5 exists at B = 4 only\n"
                 "M5 bar = 95% bootstrap CI; M3 at B = 8 exists on SUB-300 only", fontsize=9, color=INK, loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Bs, [str(b) for b in Bs])
        ax.set_xlabel("Budget B (maximum additional graph nodes)")
        ax.minorticks_off()
    axes[0].set_ylabel("ID Recall (primary)")
    axes[1].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("F1. ID Recall vs budget B", x=0.01, ha="left", fontsize=11, color=INK)
    footer(fig, "Points are slightly offset horizontally for legibility. Source: analysis/scores/aggregates.json, analysis/stats/absolute_recall_ci.json. "
                "Paired M5-vs-baseline tests: T3a (SUB-300), T1 (TEST-1704). Text Recall (diagnostic) is not plotted.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


# ------------------------------------------------------------------------------------------------------------- F2
def pareto(points):
    """Non-dominated points (lower cost and higher recall are better), sorted by cost."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1.0
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


def f2(AG):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for pi, (ax, xf, xl, title) in enumerate((
            (axes[0], "expansions_mean", "Expansions used per question (exact for every method)",
             "(a) cost = graph expansions; every (method, B)"),
            (axes[1], "total_latency_s_mean", "Wall-clock latency per question, s (log scale; exact measurements only)",
             "(b) cost = latency; only points with an exact measurement (D5)"))):
        pts = []
        for m in ("M0", "M1", "M2", "M3_mean", "M4", "M5"):
            col, mk, lab = STYLE[m]
            bs = [0] if m == "M0" else [1, 2, 4, 8]
            xy = []
            for B in bs:
                a = AG[(mkey(m, B), "SUB-300")]
                x = a[xf]
                if x is None:
                    continue
                xy.append((x, a["id_recall"], B))
            if not xy:
                continue
            sc = ax.scatter([p[0] for p in xy], [p[1] for p in xy], color=col, marker=mk, s=46 if m == "M5" else 34,
                            edgecolors=SURF, linewidths=0.8, label=lab, zorder=4)
            sc.set_gid(f"F2{'ab'[pi]}|{m}")
            for x, y, B in xy:
                if m != "M0":
                    dx, dy, ha = {"M1": (5, 3, "left"), "M2": (-6, -3, "right"), "M3_mean": (5, -9, "left"),
                                  "M4": (5, -4, "left"), "M5": (5, 4, "left")}[m]
                    ax.annotate(f"B{B}", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                                fontsize=6.5, color=INK2)
                pts.append((x, y, m, B))
                note("F2", panel="ab"[pi], series=m, B=B, x=x, x_metric=xf, id_recall=y)
        fr = pareto([(p[0], p[1], p[2], p[3]) for p in pts])
        ln, = ax.step([p[0] for p in fr], [p[1] for p in fr], where="post", color=INK2, lw=1, ls=":", label="Pareto frontier", zorder=2)
        ln.set_gid(f"F2{'ab'[pi]}|frontier")
        for p in fr:
            note("F2", panel="ab"[pi], series="frontier", B=p[3], method=p[2], x=p[0], id_recall=p[1])
        if pi == 1:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("ID Recall (primary)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("F2. ID Recall vs retrieval cost, SUB-300 (n = 300), B = 1, 2, 4, 8 (B = 0 for M0)", x=0.01, ha="left", fontsize=11, color=INK)
    footer(fig, "Each point is one (method, B); labels give B. M0 = seeds only (B = 0). Panel (b) omits M0/M1 (no added retrieval cost "
                "beyond the shared seed retrieval), M3 at B = 1/2 and M4 at B < 8 (no exact per-step latency logged). Frontier = points "
                "not beaten on both cost and recall. Source: analysis/scores/aggregates.json.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


# ------------------------------------------------------------------------------------------------------------- F3
def hop_counts(PQ, m, pop="TEST-1704"):
    out = {c: Counter() for c in CATS + ("all",)}
    for x in PQ:
        if x["method"] == m and x["B"] == 4 and x["comparison"] == "max_budget" and x["variant"] is None and x[f"in_{pop}"]:
            for k, f in (("hop0", "gold_hop0"), ("hop1", "gold_hop1"), ("hop2plus", "gold_hop2plus")):
                out[x["category"]][k] += x[f]
                out["all"][k] += x[f]
            out[x["category"]]["n_gold"] += x["n_gold"]
            out["all"]["n_gold"] += x["n_gold"]
    return out


def f3(PQ):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
    H = {m: hop_counts(PQ, m) for m in ("M4", "M5")}

    def stack(ax, x, m, cat, gid):
        bottom = 0
        for k in ("hop0", "hop1", "hop2plus"):
            v = H[m][cat][k]
            col, tcol, lab = HOP[k]
            b = ax.bar(x, v, bottom=bottom, width=0.7, color=col, edgecolor=SURF, linewidth=1.5, label=lab)
            b.patches[0].set_gid(f"{gid}|{m}|{cat}|{k}")
            if v >= 25:
                ax.text(x, bottom + v / 2, f"{v}", ha="center", va="center", fontsize=7, color=tcol)
            bottom += v
            note("F3", panel=gid, method=m, category=cat, hop=k, gold_passages=v)
        ax.text(x, bottom + 6, f"{bottom}", ha="center", va="bottom", fontsize=7.5, color=INK)
        return bottom

    ax = axes[0]
    for i, m in enumerate(("M4", "M5")):
        stack(ax, i, m, "all", "F3a")
    ax.set_xticks([0, 1], ["M4 SAGE-score\ngreedy", "M5 SLM\n(frozen)"])
    ax.set_ylabel("Gold passages in the final evidence")
    ax.set_title(f"(a) all questions\n({H['M5']['all']['n_gold']:,} gold passages in total)", fontsize=9, color=INK, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    ax = axes[1]
    xt, xl = [], []
    for ci, c in enumerate(CATS):
        for j, m in enumerate(("M4", "M5")):
            x = ci * 2.6 + j * 0.85
            stack(ax, x, m, c, "F3b")
            ax.text(x, -4, m, ha="center", va="top", fontsize=7, color=INK2)
        xt.append(ci * 2.6 + 0.425)
        xl.append(f"\n{CAT_LABEL[c]}\n(gold = {H['M5'][c]['n_gold']})")
    ax.set_xticks(xt, xl, fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_title("(b) by post-hoc category (same categories for both methods)", fontsize=9, color=INK, loc="left")
    fig.legend(handles[:3], labels[:3], loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 0.94))
    for a in axes:
        a.set_ylim(0, a.get_ylim()[1] * 1.12)
    fig.suptitle("F3. Gold passages retrieved, by hop depth, M4 vs M5 — TEST-1704 (n = 1,704), B = 4, maximum budget",
                 x=0.01, ha="left", fontsize=11, color=INK)
    footer(fig, "Hop 0 = gold already in the seeds (identical for both methods); hop 1 = discovered gold that is a direct graph "
                "neighbor of a seed; hop 2+ = discovered gold that is not. Counts of gold passages (ID-based). "
                "Source: analysis/scores/per_question.jsonl (same values as T5b).")
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    return fig


# ------------------------------------------------------------------------------------------------------------- F4
def f4(PQ):
    ranks = defaultdict(Counter)
    for x in PQ:
        if x["method"] == "M5" and x["B"] == 4 and x["comparison"] == "max_budget" and x["in_TEST-1704"]:
            for i, r in enumerate(x["selected_ranks"]):
                ranks[i + 1][r] += 1
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    ymax = max(max(c.values()) for c in ranks.values())
    for s, ax in zip((1, 2, 3, 4), axes):
        c = ranks[s]
        xs = list(range(1, 21))
        b = ax.bar(xs, [c[r] for r in xs], color=STYLE["M5"][0], edgecolor=SURF, linewidth=1, width=0.85)
        for r, p in zip(xs, b.patches):
            p.set_gid(f"F4|step{s}|rank{r}")
            note("F4", step=s, rank=r, count=c[r])
        n = sum(c.values())
        ax.set_title(f"Step {s}: {n:,} explorations\nrank 1 chosen {c[1] / n:.1%}", fontsize=8.5, color=INK, loc="left")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Pre-filter rank of the selected candidate")
        ax.set_ylim(0, ymax * 1.08)
    axes[0].set_ylabel("Explorations")
    fig.suptitle("F4. Which pre-filter rank the SLM selects, by traversal step — M5, TEST-1704 (n = 1,704), B = 4",
                 x=0.01, ha="left", fontsize=11, color=INK)
    footer(fig, "Rank = position of the chosen candidate in the SAGE pre-filter top-20 at that step. M4 (SAGE-score greedy) selects "
                "rank 1 at every step by construction. Source: selected_ranks in analysis/scores/per_question.jsonl.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


# ------------------------------------------------------------------------------------------------------------- F5
def f5(PQ, CMP):
    r5, r4 = {}, {}
    for x in PQ:
        if x["B"] == 4 and x["comparison"] == "max_budget" and x["variant"] is None and x["in_TEST-1704"]:
            if x["method"] == "M5":
                r5[x["query_id"]] = (x["id_recall"], x["category"])
            elif x["method"] == "M4":
                r4[x["query_id"]] = x["id_recall"]
    wtl = {c: Counter() for c in CATS + ("all",)}
    for q, (y5, c) in r5.items():
        d = y5 - r4[q]
        k = "win" if d > 1e-12 else "loss" if d < -1e-12 else "tie"
        wtl[c][k] += 1
        wtl["all"][k] += 1
    fig, ax = plt.subplots(figsize=(9, 4.0))
    rows = ("all",) + CATS
    for i, c in enumerate(rows):
        y = len(rows) - 1 - i
        w, l, t = wtl[c]["win"], wtl[c]["loss"], wtl[c]["tie"]
        bw = ax.barh(y, w, color=WIN, height=0.6, edgecolor=SURF, linewidth=1.5, label="M5 higher recall (win)" if i == 0 else None)
        bl = ax.barh(y, -l, color=LOSS, height=0.6, edgecolor=SURF, linewidth=1.5, label="M4 higher recall (loss)" if i == 0 else None)
        bw.patches[0].set_gid(f"F5|{c}|win")
        bl.patches[0].set_gid(f"F5|{c}|loss")
        ax.text(w + 2, y, f"{w}", va="center", fontsize=7.5, color=INK)
        ax.text(-l - 2, y, f"{l}", va="center", ha="right", fontsize=7.5, color=INK)
        ax.text(1.0, y, f"ties {t}  (n = {w + l + t})", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.5, color=INK2)
        note("F5", category=c, wins=w, ties=t, losses=l)
    ax.set_yticks(range(len(rows)), [("All questions" if c == "all" else CAT_LABEL[c].replace("\n", " ")) for c in rows][::-1])
    ax.axvline(0, color=AXIS, lw=1)
    lim = max(max(wtl[c]["win"], wtl[c]["loss"]) for c in rows) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Questions (losses to the left, wins to the right; ties listed at right)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=7.5)
    c = CMP[(ck("M4"), ck("M5"), "TEST-1704")]
    fig.suptitle("F5. Per-question paired outcome, M5 SLM vs M4 SAGE-score greedy — TEST-1704 (n = 1,704), B = 4, maximum budget",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    footer(fig, f"Win/tie/loss on per-question ID Recall (the T1 comparison: Δ = {c['mean_diff']:+.4f}, 95% CI "
                f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], Holm p {'<0.0001' if c['p_sign_holm'] < 1e-4 else round(c['p_sign_holm'], 4)}). "
                "Categories as in T5c. Source: analysis/scores/per_question.jsonl, analysis/stats/comparisons.json.")
    fig.tight_layout(rect=(0, 0.06, 0.86, 0.93))
    return fig, wtl


# ------------------------------------------------------------------------------------------------------------- validation
def by_gid(fig):
    out = {}
    for ax in fig.axes:
        for a in list(ax.lines) + list(ax.collections) + list(ax.patches):
            if a.get_gid():
                out[a.get_gid()] = a
    return out


def validate(figs, AG, ABS, CMP, PQ, wtl):
    v, tol = {}, 1e-12

    def chk(n, c, **d):
        v[n] = {"PASS": bool(c), **d}

    G = by_gid(figs["F1"])
    bad = 0
    for pan, pop in (("a", "SUB-300"), ("b", "TEST-1704")):
        for m in ("M1", "M2", "M3_mean", "M4", "M5"):
            ln = G.get(f"F1{pan}|line|{m}")
            if ln is None:
                bad += 1
                continue
            bs = [4] if (pan == "b" and m == "M5") else [B for B in (1, 2, 4, 8) if (mkey(m, B), pop) in AG]
            ys = [AG[(mkey(m, B), pop)]["id_recall"] for B in bs]
            bad += any(abs(a - b) > tol for a, b in zip(ln.get_ydata(), ys)) or len(ln.get_ydata()) != len(ys)
            bad += any(abs(x / B - 1) > 0.05 for x, B in zip(ln.get_xdata(), bs))   # offset only, same budget
        bad += abs(G[f"F1{pan}|M0"].get_ydata()[0] - AG[(mkey("M0", 0), pop)]["id_recall"]) > tol
    for m in ("M1", "M2", "M3_mean", "M4", "M5"):
        segs = G[f"F1a|ci|{m}"].get_segments()
        cis = [ABS[(mkey(m, B), "SUB-300")]["ci95"] for B in (1, 2, 4, 8)]
        bad += any(abs(s[0][1] - c[0]) > tol or abs(s[1][1] - c[1]) > tol for s, c in zip(segs, cis))
    for pan, pop in (("a", "SUB-300"), ("b", "TEST-1704")):
        band = G[f"F1{pan}|band|M3_mean"]
        ys = band.get_paths()[0].vertices[:, 1]
        bs = [B for B in (1, 2, 4, 8) if (mkey("M3_mean", B), pop) in AG]
        exp = [AG[(mkey("M3_mean", B), pop)]["id_recall"] + s * AG[(mkey("M3_mean", B), pop)]["id_recall_seed_sd"] for B in bs for s in (-1, 1)]
        bad += abs(min(ys) - min(exp)) > tol or abs(max(ys) - max(exp)) > tol
    chk("F1: every plotted recall, CI and M3 ±SD band equals the canonical source", bad == 0, mismatches=bad)

    bad, fr_bad = 0, 0
    for pi, xf in (("a", "expansions_mean"), ("b", "total_latency_s_mean")):
        G = by_gid(figs["F2"])
        pts = []
        for m in ("M0", "M1", "M2", "M3_mean", "M4", "M5"):
            exp = [(AG[(mkey(m, B), "SUB-300")][xf], AG[(mkey(m, B), "SUB-300")]["id_recall"]) for B in ([0] if m == "M0" else (1, 2, 4, 8))
                   if AG[(mkey(m, B), "SUB-300")][xf] is not None]
            sc = G.get(f"F2{pi}|{m}")
            got = [] if sc is None else [tuple(o) for o in sc.get_offsets()]
            bad += len(got) != len(exp) or any(abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol for a, b in zip(got, exp))
            pts += exp
        fx, fy = G[f"F2{pi}|frontier"].get_xdata(), G[f"F2{pi}|frontier"].get_ydata()
        front = set(zip(fx, fy))
        nd = {p for p in pts if not any((q[0] <= p[0] and q[1] > p[1]) or (q[0] < p[0] and q[1] >= p[1]) for q in pts)}
        fr_bad += front != nd
    chk("F2: every plotted (cost, recall) point equals the canonical source", bad == 0, mismatches=bad)
    chk("F2: frontier = exactly the non-dominated points (independent check)", fr_bad == 0)

    G = by_gid(figs["F3"])
    agg_ok = True
    for m in ("M4", "M5"):
        a = AG[(ck(m), "TEST-1704")]
        h = {k: G[f"F3a|{m}|all|{k}"].get_height() for k in ("hop0", "hop1", "hop2plus")}
        agg_ok &= h["hop1"] == a["gold_hop1_total"] and h["hop2plus"] == a["gold_hop2plus_total"]
        hop0 = sum(x["gold_hop0"] for x in PQ if x["method"] == "M0" and x["in_TEST-1704"])
        agg_ok &= h["hop0"] == hop0
        agg_ok &= all(sum(G[f"F3b|{m}|{c}|{k}"].get_height() for c in CATS) == h[k] for k in h)
    chk("F3: hop-1 and hop-2+ bars equal the canonical aggregates (= T5b); hop-0 equals the seeds' gold; categories sum to totals", agg_ok)
    chk("F3: hop 0 identical for M4 and M5 in every category", all(
        G[f"F3b|M4|{c}|hop0"].get_height() == G[f"F3b|M5|{c}|hop0"].get_height() for c in CATS))
    chk("F3: 'all gold in seeds' category has no discovered gold", all(
        G[f"F3b|{m}|all_gold_in_seeds|{k}"].get_height() == 0 for m in ("M4", "M5") for k in ("hop1", "hop2plus")))

    G = by_gid(figs["F4"])
    tot = sum(G[f"F4|step{s}|rank{r}"].get_height() for s in range(1, 5) for r in range(1, 21))
    exp_tot = sum(x["expansions"] for x in PQ if x["method"] == "M5" and x["B"] == 4 and x["comparison"] == "max_budget" and x["in_TEST-1704"])
    agg = AG[(ck("M5"), "TEST-1704")]
    chk("F4: total explorations plotted = total M5 expansions on TEST-1704 (= mean expansions × n)",
        tot == exp_tot and abs(exp_tot - agg["expansions_mean"] * agg["n"]) < 1e-6, plotted=int(tot), expected=int(exp_tot))
    chk("F4: no rank outside 1..20", all(r in range(1, 21) for x in PQ if x["method"] == "M5" and x["B"] == 4
                                         and x["comparison"] == "max_budget" for r in x["selected_ranks"]))

    G = by_gid(figs["F5"])
    c = CMP[(ck("M4"), ck("M5"), "TEST-1704")]
    allw, alll = G["F5|all|win"].get_width(), -G["F5|all|loss"].get_width()
    chk("F5: 'All questions' W/T/L equals the frozen T1 comparison (comparisons.json)",
        (allw, wtl["all"]["tie"], alll) == (c["wins"], c["ties"], c["losses"]), plotted=[int(allw), wtl["all"]["tie"], int(alll)],
        canonical=[c["wins"], c["ties"], c["losses"]])
    chk("F5: categories sum to the all-questions bar", all(sum(wtl[k][o] for k in CATS) == wtl["all"][o] for o in ("win", "tie", "loss")))
    labels_ok = True
    for fid, fig in figs.items():
        txt = " ".join(t.get_text() for t in fig.texts) + " " + " ".join(ax.get_title(loc) for ax in fig.axes for loc in ("left", "center")) \
            + " " + (fig._suptitle.get_text() if fig._suptitle else "")
        labels_ok &= any(p in txt for p in ("SUB-300", "TEST-1704")) and "B =" in txt
    chk("every figure states its population and budget", labels_ok)
    chk("ID Recall labelled primary on every recall axis",
        all("ID Recall (primary)" in figs[f].axes[0].get_ylabel() for f in ("F1", "F2")))
    v["ALL_PASS"] = all(x["PASS"] for x in v.values() if isinstance(x, dict))
    return v


def main():
    verify()
    t_before = {p.name: sha(p) for p in (PROJECT / "analysis/tables").iterdir() if p.is_file()}
    AG, ABS, CMP, PQ = load()
    figs = {"F1": f1(AG, ABS), "F2": f2(AG), "F3": f3(PQ), "F4": f4(PQ)}
    figs["F5"], wtl = f5(PQ, CMP)
    for fig in figs.values():
        fig.canvas.draw()
    val = validate(figs, AG, ABS, CMP, PQ, wtl)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, fig in figs.items():
        fig.savefig(OUT / f"{k}.png", dpi=200)
        fig.savefig(OUT / f"{k}.pdf", metadata={"CreationDate": None, "ModDate": None})
        rows = DATA[k]
        cols = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        with open(OUT / f"{k}_data.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    val["T1–T6 files unchanged"] = {"PASS": t_before == {p.name: sha(p) for p in (PROJECT / "analysis/tables").iterdir() if p.is_file()}}
    val["ALL_PASS"] = all(x["PASS"] for x in val.values() if isinstance(x, dict))
    (OUT / "figures_validation.json").write_text(json.dumps(val, indent=1, default=lambda o: o.item()))
    (OUT / "figures_manifest.json").write_text(json.dumps({"figures_py_sha256": sha(Path(__file__)),
        "inputs_sha256": {**json.loads((SC / "scoring_manifest.json").read_text())["outputs_sha256"],
                          **json.loads((ST / "stats_manifest.json").read_text())["outputs_sha256"]},
        "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.suffix in (".csv", ".json") and p.name != "figures_manifest.json"}},
        indent=1))
    for k, x in val.items():
        if isinstance(x, dict):
            print(("PASS " if x["PASS"] else "FAIL ") + k)
    print("ALL_PASS" if val["ALL_PASS"] else "FAILED")


if __name__ == "__main__":
    main()
