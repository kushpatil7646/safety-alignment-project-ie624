"""
Evaluation script: runs the full detection pipeline on all models in config,
trains/evaluates the Stage 4 classifier, compares against baselines,
and saves a comprehensive results JSON + plots.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_curve, auc

from .pipeline import load_config, run_pipeline_for_model, build_clean_reference
from .probe_generator import generate_probe_pairs
from .stage4_classifier import (
    BackdoorClassifier,
    aggregate_scores,
    cross_validate,
    evaluate_classifier,
)
from .baselines import ONIONDetector, RandomBaseline

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def plot_roc_curves(results_by_method: dict, output_path: str):
    plt.figure(figsize=(8, 6))
    for method_name, (labels, scores) in results_by_method.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{method_name} (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Backdoor Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")


def plot_score_distributions(model_results: list[dict], output_path: str):
    clean = [r for r in model_results if r["label"] == 0]
    backdoored = [r for r in model_results if r["label"] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, score_key, title in zip(axes, ["s1", "s2", "s3"],
                                    ["Stage 1: KL Divergence Score",
                                     "Stage 2: Representation Score",
                                     "Stage 3: Trigger Search Score"]):
        c_vals = [r[score_key] for r in clean if score_key in r]
        b_vals = [r[score_key] for r in backdoored if score_key in r]
        if c_vals:
            ax.hist(c_vals, bins=15, alpha=0.6, label="Clean", color="steelblue")
        if b_vals:
            ax.hist(b_vals, bins=15, alpha=0.6, label="Backdoored", color="tomato")
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Score distributions saved to {output_path}")


def run_evaluation(config_path: str, use_test_models: bool = False, skip_stage3: bool = False, laptop_mode: bool = False, skip_stages: set = None):
    config = load_config(config_path)

    if laptop_mode:
        overrides = config.get("laptop", {})
        logger.info("Laptop mode enabled — applying low-resource overrides.")
        for section, values in overrides.items():
            if section in ("stage1", "stage2", "stage3", "probe"):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values

    output_dir = config.get("evaluation", {}).get("output_dir", "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    random.seed(config.get("stage4", {}).get("random_seed", 42))

    # ── Generate probe set ────────────────────────────────────────────────────
    logger.info("Generating probe pairs ...")
    probe_cfg = config.get("probe", {})
    probe_pairs = generate_probe_pairs(
        domains=probe_cfg.get("domains", ["news", "dialogue", "sentiment", "qa"]),
        n_base_per_domain=probe_cfg.get("n_base_inputs", 20),
        n_perturbations=probe_cfg.get("n_perturbations", 4),
        random_seed=config.get("stage4", {}).get("random_seed", 42),
    )
    logger.info(f"Generated {len(probe_pairs)} probe pairs.")

    # ── Select model list ─────────────────────────────────────────────────────
    if use_test_models:
        clean_models = config["models"].get("test_clean", ["gpt2"])
        backdoored_models = config["models"].get("test_backdoored", [])
    else:
        clean_models = config["models"]["clean"]
        backdoored_models = config["models"]["backdoored"]

    all_models = [(m, 0) for m in clean_models] + [(m, 1) for m in backdoored_models]
    logger.info(f"Models to evaluate: {len(clean_models)} clean, {len(backdoored_models)} backdoored")

    # ── Build clean reference (use first clean model) ─────────────────────────
    clean_reference = None
    if clean_models:
        logger.info(f"Building clean reference from: {clean_models[0]}")
        try:
            clean_reference = build_clean_reference(clean_models[0], probe_pairs[:50], config)
            logger.info("Clean reference built.")
        except Exception as e:
            logger.warning(f"Could not build clean reference: {e}")

    # ── Run all stages for each model ─────────────────────────────────────────
    model_results = []
    for model_name, label in all_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {model_name} (label={'backdoored' if label else 'clean'})")
        result = run_pipeline_for_model(
            model_name=model_name,
            label=label,
            config=config,
            probe_pairs=probe_pairs,
            clean_reference=clean_reference,
            output_dir=output_dir,
            skip_stages=skip_stages,
        )
        model_results.append(result)

    # Save all model results
    with open(f"{output_dir}/all_model_results.json", "w") as f:
        json.dump(model_results, f, indent=2, default=str)

    if len(model_results) < 2:
        logger.warning("Not enough model results for classifier training/evaluation.")
        return model_results, {}

    # ── Stage 4: Classifier ───────────────────────────────────────────────────
    features, labels = aggregate_scores(model_results)
    logger.info(f"\nStage 4: Training classifier on {len(features)} models ...")

    classifier = BackdoorClassifier(random_seed=config.get("stage4", {}).get("random_seed", 42))
    classifier.fit(features, labels)

    if len(np.unique(labels)) > 1:
        classifier.set_threshold_at_tpr(features, labels, tpr=0.95)

    eval_results = evaluate_classifier(classifier, features, labels)

    # Cross-validation
    cv_results = {}
    if len(features) >= 6 and len(np.unique(labels)) > 1:
        cv_results = cross_validate(features, labels, n_folds=min(5, len(features) // 2))

    classifier.save(f"{output_dir}/classifier.pkl")

    # ── Baselines ─────────────────────────────────────────────────────────────
    logger.info("\nRunning baselines ...")
    baseline_results = {}

    # ONION
    try:
        onion = ONIONDetector(device="cpu")
        onion_scores = []
        for model_name, label in all_models:
            probe_texts = list({p.base for p in probe_pairs})[:30]
            score = onion.score_model(probe_texts, threshold=config["baselines"]["onion"]["perplexity_threshold"])
            onion_scores.append(score)
        baseline_results["ONION"] = (labels, np.array(onion_scores))
        logger.info(f"ONION scores computed.")
    except Exception as e:
        logger.warning(f"ONION baseline failed: {e}")

    # Activation Clustering — reuse Stage 2 bimodality scores (same GMM, no extra model load)
    try:
        ac_scores = []
        for r in model_results:
            bimod = r.get("diagnostics", {}).get("stage2", {}).get("mean_bimodality", 0.0)
            ac_scores.append(float(bimod))
        baseline_results["ActivationClustering"] = (labels, np.array(ac_scores))
        logger.info("AC scores extracted from Stage 2 diagnostics.")
    except Exception as e:
        logger.warning(f"AC baseline failed: {e}")

    # Random
    rb = RandomBaseline(majority_class=1)
    baseline_results["Random"] = (labels, rb.predict_proba(len(labels)))

    # Our method
    our_proba = classifier.predict_proba(features)
    baseline_results["OurMethod"] = (labels, our_proba)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(baseline_results, f"{output_dir}/roc_curves.png")
    plot_score_distributions(model_results, f"{output_dir}/score_distributions.png")

    # ── Baseline metrics ──────────────────────────────────────────────────────
    baseline_metrics = {}
    for method, (lbls, scores) in baseline_results.items():
        if len(np.unique(lbls)) < 2:
            continue
        from sklearn.metrics import roc_auc_score, accuracy_score
        preds = (scores >= 0.5).astype(int)
        fpr_arr, tpr_arr, _ = roc_curve(lbls, scores)
        idx = np.searchsorted(tpr_arr, 0.95)
        fpr95 = float(fpr_arr[idx]) if idx < len(fpr_arr) else float("nan")
        baseline_metrics[method] = {
            "auroc": float(roc_auc_score(lbls, scores)),
            "fpr_at_95tpr": fpr95,
            "accuracy": float(accuracy_score(lbls, preds)),
        }

    # ── Final summary ─────────────────────────────────────────────────────────
    summary = {
        "our_method": {**eval_results, **cv_results},
        "baselines": baseline_metrics,
        "n_models": len(model_results),
        "n_clean": int((labels == 0).sum()),
        "n_backdoored": int((labels == 1).sum()),
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Our Method:  AUROC={eval_results['auroc']:.4f}  FPR@95TPR={eval_results['fpr_at_95tpr']:.4f}  Acc={eval_results['accuracy']:.4f}")
    for m, v in baseline_metrics.items():
        if m != "OurMethod":
            logger.info(f"{m:25s}: AUROC={v['auroc']:.4f}  FPR@95TPR={v['fpr_at_95tpr']:.4f}  Acc={v['accuracy']:.4f}")

    return model_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--test", action="store_true", help="Use small test models (gpt2) for quick run")
    parser.add_argument("--laptop", action="store_true", help="Apply low-resource overrides for laptop use")
    parser.add_argument("--skip-stages", type=str, default="", help="Comma-separated stages to skip, e.g. '1,3'")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    skip_stages = {int(s) for s in args.skip_stages.split(",") if s.strip()} if args.skip_stages else set()

    setup_logging(args.log_level)
    config = None
    if args.output_dir:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        cfg.setdefault("evaluation", {})["output_dir"] = args.output_dir
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tmp)
        tmp.close()
        run_evaluation(tmp.name, use_test_models=args.test, laptop_mode=args.laptop, skip_stages=skip_stages)
        os.unlink(tmp.name)
    else:
        run_evaluation(args.config, use_test_models=args.test, laptop_mode=args.laptop, skip_stages=skip_stages)
