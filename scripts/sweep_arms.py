"""Sample-efficiency sweep: baseline vs grammar heads at small data, several seeds — the test the hypothesis actually makes.

  python scripts/sweep_arms.py --config configs/tiny.json --tokens 250000 1000000 4000000 --seeds 1 2 3 \
      --arms none upos chunk --blimp external/blimp/data --blimp_limit 100 --out results/sweep.json

The CPU n-gram run showed the POS-class gain shrinking as data grows (-13.6% perplexity at 100k tokens, -8% at 3.5M);
the neural comparison in run 1 was made after 33M tokens, where any such gain would be gone. This sweep trains the
tiny model for exactly N tokens per arm and seed, evaluates bits/byte on gold GUM (+ BLiMP), deletes the checkpoints,
and plots mean with min-max whiskers per arm. If "upos" beats "none" by more than the seed spread at 250k-1M tokens
and the arms converge at 4M, grammar buys data efficiency at zero inference cost; if not, the variant is dead.
"""
import argparse, json, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ARMS = {"none": "[]", "upos": '["upos"]', "chunk": '["chunk"]', "both": '["upos","chunk"]'}
COLORS = {"none": "#2a78d6", "upos": "#eb6834", "chunk": "#1baf7a", "both": "#eda100"}


def run(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.json")
    ap.add_argument("--tokens", type=int, nargs="+", default=[250000, 1000000, 4000000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=["none", "upos", "chunk"])
    ap.add_argument("--blimp", default=None)
    ap.add_argument("--blimp_limit", type=int, default=100)
    ap.add_argument("--out", default="results/sweep.json")
    ap.add_argument("--plot", default="results/sweep.png")
    ap.add_argument("--keep", action="store_true", help="keep the checkpoints")
    args = ap.parse_args()
    from gramlab.config import load_config, run_dir
    base = load_config(args.config)
    tok_per_step = base["train"]["batch"] * base["train"]["accum"] * base["model"]["ctx"]
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r["arm"], r["tokens"], r["seed"]) for r in results}
    t0 = time.time()
    for n in args.tokens:
        steps = max(10, n // tok_per_step)
        for arm in args.arms:
            for seed in args.seeds:
                if (arm, n, seed) in done:
                    continue
                name = f"sweep_{arm}_{n // 1000}k_s{seed}"
                sets = (f"--set run_name={name} --set train.seed={seed} --set train.steps={steps} --set train.eval_every={steps} "
                        f"--set train.warmup={max(5, steps // 10)} --set train.eval_tokens=200000 --set 'model.aux_heads={ARMS[arm]}'")
                r = run(f"{sys.executable} -m gramlab.train --config {args.config} {sets}")
                if r.returncode != 0:
                    print(r.stderr[-1500:]); continue
                rd = run_dir(load_config(args.config, [f"run_name={name}"]))
                ev = json.loads(open(os.path.join(rd, "log.jsonl")).read().strip().splitlines()[-1])
                rec = {"arm": arm, "tokens": n, "seed": seed, "steps": steps, "bits_per_byte": ev["eval_bits_per_byte"], "eval_ppl": ev["eval_ppl"]}
                if args.blimp:
                    e = run(f"{sys.executable} -m gramlab.evaluate --ckpt {rd}/ckpt.pt --blimp {args.blimp} --blimp_limit {args.blimp_limit}")
                    if e.returncode == 0:
                        rec["blimp"] = json.loads(e.stdout[e.stdout.index("{"):])["blimp"]["_mean"]
                results.append(rec)
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                json.dump(results, open(args.out, "w"), indent=1)
                print(f"[sweep] {name}: bits/byte {rec['bits_per_byte']:.4f}" + (f"  BLiMP {rec['blimp']:.4f}" if "blimp" in rec else "")
                      + f"  ({(time.time() - t0) / 60:.0f} min)", flush=True)
                if not args.keep:
                    shutil.rmtree(rd, ignore_errors=True)
    # ---- summary table + plot
    import numpy as np
    print("\narm      tokens    bits/byte mean (min-max)" + ("   BLiMP mean" if args.blimp else ""))
    for n in args.tokens:
        for arm in args.arms:
            v = [r["bits_per_byte"] for r in results if r["arm"] == arm and r["tokens"] == n]
            b = [r["blimp"] for r in results if r["arm"] == arm and r["tokens"] == n and "blimp" in r]
            if v:
                print(f"{arm:8} {n:>9,}  {np.mean(v):.4f} ({min(v):.4f}-{max(v):.4f})" + (f"   {np.mean(b):.4f}" if b else ""))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("bits_per_byte", "bits/byte on gold GUM (lower is better)")] + ([("blimp", "BLiMP accuracy (higher is better)")] if args.blimp else [])
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.2), facecolor="white", squeeze=False)
    for ax, (key, label) in zip(axes[0], metrics):
        for arm in args.arms:
            xs, ms, lo, hi = [], [], [], []
            for n in args.tokens:
                v = [r[key] for r in results if r["arm"] == arm and r["tokens"] == n and key in r]
                if v:
                    xs.append(n); ms.append(np.mean(v)); lo.append(np.mean(v) - min(v)); hi.append(max(v) - np.mean(v))
            if xs:
                ax.errorbar(xs, ms, yerr=[lo, hi], color=COLORS.get(arm, "#52514e"), marker="o", ms=5, lw=2, capsize=3, label=f"aux heads: {arm}")
        ax.set_xscale("log"); ax.set_xlabel("training tokens"); ax.set_ylabel(label)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color="#e5e4e0"); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=9)
    axes[0][0].set_title(f"Grammar heads vs baseline, {len(args.seeds)} seeds (whiskers = min-max)", loc="left", fontsize=11)
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    fig.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"[sweep] -> {args.out}, {args.plot}")


if __name__ == "__main__":
    main()
