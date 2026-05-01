"""
Post-experiment analysis: load results and produce all plots + tables.
Run after run_experiment.sh completes:
    python notebooks/analysis.py --results results/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def load_results(results_dir: str):
    p = Path(results_dir)
    with open(p / "all_model_results.json") as f:
        model_results = json.load(f)
    with open(p / "summary.json") as f:
        summary = json.load(f)
    return model_results, summary


def print_table(summary: dict):
    print("\n" + "=" * 70)
    print(f"{'Method':<28} {'AUROC':>8} {'FPR@95TPR':>10} {'Accuracy':>10}")
    print("-" * 70)
    ours = summary.get("our_method", {})
    print(f"{'Our Method (4-Stage)':<28} {ours.get('auroc', float('nan')):>8.4f} "
          f"{ours.get('fpr_at_95tpr', float('nan')):>10.4f} "
          f"{ours.get('accuracy', float('nan')):>10.4f}")
    for name, m in summary.get("baselines", {}).items():
        if name == "OurMethod":
            continue
        print(f"{name:<28} {m.get('auroc', float('nan')):>8.4f} "
              f"{m.get('fpr_at_95tpr', float('nan')):>10.4f} "
              f"{m.get('accuracy', float('nan')):>10.4f}")
    print("=" * 70)
    cv = ours.get("cv_auroc")
    if cv:
        print(f"\nCross-validated AUROC: {cv:.4f}")


def plot_ablation(model_results: list, output_dir: str):
    """Ablation: test performance with subsets of scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    labels = np.array([r["label"] for r in model_results])
    if len(np.unique(labels)) < 2:
        print("Skipping ablation: need both classes.")
        return

    feature_sets = {
        "s1 only": [0],
        "s2 only": [1],
        "s3 only": [2],
        "s1+s2": [0, 1],
        "s1+s3": [0, 2],
        "s2+s3": [1, 2],
        "s1+s2+s3": [0, 1, 2],
    }

    aurocs = {}
    all_feats = np.array([[r["s1"], r["s2"], r["s3"]] for r in model_results])

    for name, idxs in feature_sets.items():
        X = all_feats[:, idxs]
        if len(X) < 4:
            continue
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000).fit(X_s, labels)
        proba = clf.predict_proba(X_s)[:, 1]
        aurocs[name] = roc_auc_score(labels, proba)

    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(aurocs.keys())
    vals = list(aurocs.values())
    bars = ax.bar(names, vals, color="steelblue", alpha=0.8)
    ax.axhline(0.9, color="green", linestyle="--", label="Target AUROC (0.90)")
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation Study: Score Component Contribution")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation.png", dpi=150)
    plt.close()
    print(f"Ablation plot saved to {output_dir}/ablation.png")


def plot_per_attack_type(model_results: list, output_dir: str):
    """Bar chart of detection score per attack type (from model name)."""
    attack_types = {}
    for r in model_results:
        name = r["model"]
        # Parse attack type from BackdoorLLM model name pattern
        if "BadWords" in name:
            atype = "BadWords"
        elif "BadSent" in name:
            atype = "BadSent"
        elif "StyleBkd" in name:
            atype = "StyleBkd"
        elif "VPI" in name:
            atype = "VPI"
        elif "clean" in name.lower():
            atype = "Clean"
        else:
            atype = "Other"

        score = r.get("s1", 0) + r.get("s2", 0) + r.get("s3", 0)
        attack_types.setdefault(atype, []).append(score)

    if not attack_types:
        return

    labels = list(attack_types.keys())
    means = [np.mean(v) for v in attack_types.values()]
    stds = [np.std(v) for v in attack_types.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["steelblue" if l == "Clean" else "tomato" for l in labels]
    ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel("Combined Anomaly Score (s1+s2+s3)")
    ax.set_title("Detection Score by Attack Type")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/per_attack_type.png", dpi=150)
    plt.close()
    print(f"Per-attack-type plot saved to {output_dir}/per_attack_type.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    model_results, summary = load_results(args.results)

    print(f"\nLoaded {len(model_results)} model results.")
    print_table(summary)

    plot_ablation(model_results, args.results)
    plot_per_attack_type(model_results, args.results)

    print(f"\nAll analysis plots saved to {args.results}/")


if __name__ == "__main__":
    main()
