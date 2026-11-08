"""Per-POS loss chart from `python -m gramlab.evaluate --per_pos --out results/eval_<run>.json` files.

  python scripts/plot_pos_loss.py --results results/eval_tiny.json results/eval_tiny_upos.json results/eval_small_wiki.json \
                                  --out results/per_pos_loss.png

Top: mean next-token loss (nats) by the UPOS tag of the token being predicted, tags ordered by how much of the
text they make up (share under each label). Bottom left: loss at chunk starts (B-*) vs inside chunks (I-*) vs
outside any chunk (O). Bottom right (two or more runs): change versus the first run per tag; below zero = better.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
BETTER, WORSE, GRID, INK, INK2 = "#2a78d6", "#e34948", "#e5e4e0", "#0b0b0b", "#52514e"
POS_LABEL = {"B": "chunk start (B-*)", "I": "inside chunk (I-*)", "O": "outside chunks (O)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="results/per_pos_loss.png")
    ap.add_argument("--min_share", type=float, default=0.004, help="hide tags rarer than this share of tokens")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    runs = [json.load(open(p)) for p in args.results]
    labels = args.labels or [r.get("run_name") or os.path.basename(p).replace("eval_", "").replace(".json", "") for r, p in zip(runs, args.results)]
    for r in runs:
        if "per_upos" not in r:
            sys.exit(f"{r.get('ckpt')}: no per_upos data; run evaluate with --per_pos on a tagged corpus")
    base = runs[0]["per_upos"]
    tags = [t for t in sorted(base, key=lambda t: -base[t]["share"]) if base[t]["share"] >= args.min_share and t != "EOS"]
    n, m = len(tags), len(runs)
    x = np.arange(n)
    width = min(0.8 / m, 0.28)

    fig = plt.figure(figsize=(max(9, 0.75 * n + 3), 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1, 1.6], hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    for k, (r, lab) in enumerate(zip(runs, labels)):
        vals = [r["per_upos"].get(t, {}).get("loss", np.nan) for t in tags]
        ax.bar(x + (k - (m - 1) / 2) * (width + 0.02), vals, width, color=SERIES[k % len(SERIES)],
               label=f"{lab}  (all tagged tokens: {r.get('tagged_loss', float('nan')):.2f})", linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{100 * base[t]['share']:.0f}%" for t in tags], color=INK)
    ax.set_ylabel("next-token loss (nats)", color=INK2)
    ax.set_title(args.title or "Loss by part of speech of the predicted token", loc="left", color=INK, fontsize=12, pad=12)
    if m > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    else:
        ax.text(0.99, 0.97, f"{labels[0]}  (all tagged tokens: {runs[0].get('tagged_loss', float('nan')):.2f})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=INK2)

    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2)
    keys = [k for k in ("B", "I", "O") if all(k in r.get("per_chunk_position", {}) for r in runs)]
    xp = np.arange(len(keys))
    for k, r in enumerate(runs):
        ax2.bar(xp + (k - (m - 1) / 2) * (width + 0.02), [r["per_chunk_position"][q]["loss"] for q in keys], width,
                color=SERIES[k % len(SERIES)], linewidth=0)
    ax2.set_xticks(xp)
    ax2.set_xticklabels([POS_LABEL[q] for q in keys], color=INK, fontsize=8.5)
    ax2.set_ylabel("loss (nats)", color=INK2)
    ax2.set_title("Chunk position of the predicted token", loc="left", color=INK, fontsize=10.5)

    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3)
    if m > 1:
        wd = min(0.8 / (m - 1), 0.4)
        for k in range(1, m):
            d = np.array([runs[k]["per_upos"].get(t, {}).get("loss", np.nan) - base[t]["loss"] for t in tags])
            # one comparison: colour by sign (blue better / red worse); several: keep each run's own colour
            colors = np.where(d <= 0, BETTER, WORSE) if m == 2 else SERIES[k % len(SERIES)]
            ax3.bar(x + (k - 1 - (m - 2) / 2) * (wd + 0.02), d, wd, color=colors, linewidth=0, label=labels[k])
            for xi, di in zip(x, d):                          # label the largest few changes only
                if abs(di) >= np.nanmax(np.abs(d)) * 0.6:
                    ax3.text(xi + (k - 1 - (m - 2) / 2) * (wd + 0.02), di + (0.004 if di >= 0 else -0.004), f"{di:+.2f}",
                             ha="center", va="bottom" if di >= 0 else "top", fontsize=7.5, color=INK2)
        ax3.axhline(0, color=INK2, linewidth=0.8)
        if m > 2:
            ax3.legend(frameon=False, fontsize=8.5, loc="lower left")
        ax3.set_xticks(x)
        ax3.set_xticklabels(tags, color=INK, fontsize=8, rotation=45, ha="right")
        ax3.set_ylabel(f"Δ loss vs {labels[0]} (nats)", color=INK2)
        who = ", ".join(labels[1:])
        ax3.set_title(f"{who} vs {labels[0]}: below zero = better", loc="left", color=INK, fontsize=10.5)
    else:
        ax3.axis("off")
        ax3.text(0, 0.5, "Add a second results file to see per-tag deltas.", color=INK2, fontsize=9)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot] {args.out}")
    print("tag      share  " + "  ".join(f"{l[:14]:>14}" for l in labels))
    for t in tags:
        print(f"{t:8} {100 * base[t]['share']:5.1f}%  " + "  ".join(f"{r['per_upos'].get(t, {}).get('loss', float('nan')):14.3f}" for r in runs))


if __name__ == "__main__":
    main()
