"""
Main detection pipeline: orchestrates all 4 stages for a single model
and returns per-model feature vector (s1, s2, s3) + diagnostics.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .model_loader import load_model_and_tokenizer
from .probe_generator import generate_probe_pairs, ProbePair
from .stage1_behavioral import run_stage1
from .stage2_representation import compute_s2_score
from .stage3_trigger_search import compute_s3_score

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline_for_model(
    model_name: str,
    label: int,  # 0=clean, 1=backdoored
    config: dict,
    probe_pairs: list[ProbePair],
    clean_reference: Optional[dict] = None,
    output_dir: str = "results",
    skip_stages: Optional[set] = None,
) -> dict:
    """
    Run all 4 stages for a single model.
    Returns dict with keys: model, label, s1, s2, s3, diagnostics, runtime_s.
    """
    t0 = time.time()
    result = {"model": model_name, "label": label}

    cache_dir = config.get("hf_cache_dir")
    device = config.get("device", "auto")
    dtype_str = config.get("dtype", "float16")

    try:
        model, tokenizer = load_model_and_tokenizer(model_name, cache_dir, device, dtype_str)
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        result.update({"s1": 0.0, "s2": 0.0, "s3": 0.0, "error": str(e)})
        return result

    cfg1 = config.get("stage1", {})
    cfg2 = config.get("stage2", {})
    cfg3 = config.get("stage3", {})
    max_length = config.get("max_length", 512)
    skip = skip_stages or set()

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if 1 not in skip:
        try:
            s1, diag1 = run_stage1(
                model, tokenizer, probe_pairs,
                top_k=cfg1.get("kl_top_k", 50),
                tail_pct=cfg1.get("tail_threshold_pct", 95),
                batch_size=cfg1.get("batch_size", 16),
                clean_reference=clean_reference.get("kl_by_type") if clean_reference else None,
                max_length=max_length,
            )
        except Exception as e:
            logger.error(f"Stage 1 failed for {model_name}: {e}")
            s1, diag1 = 0.0, {"error": str(e)}
    else:
        s1, diag1 = 0.0, {"skipped": True}

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if 2 not in skip:
        try:
            s2, diag2 = compute_s2_score(
                model, tokenizer, probe_pairs,
                layer_indices=cfg2.get("layers", [-1]),
                batch_size=cfg2.get("batch_size", 16),
                clean_reference_score=clean_reference.get("bimodality") if clean_reference else None,
                max_length=max_length,
            )
        except Exception as e:
            logger.error(f"Stage 2 failed for {model_name}: {e}")
            s2, diag2 = 0.0, {"error": str(e)}
    else:
        s2, diag2 = 0.0, {"skipped": True}

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if 3 not in skip:
        try:
            s3, diag3 = compute_s3_score(
                model, tokenizer, probe_pairs,
                top_vocab_tokens=cfg3.get("top_vocab_tokens", 100),
                n_search_inputs=cfg3.get("n_search_inputs", 10),
                top_k=cfg1.get("kl_top_k", 50),
                batch_size=cfg1.get("batch_size", 16),
                clean_reference_score=clean_reference.get("trigger_score") if clean_reference else None,
                max_length=max_length,
            )
        except Exception as e:
            logger.error(f"Stage 3 failed for {model_name}: {e}")
            s3, diag3 = 0.0, {"error": str(e)}
    else:
        s3, diag3 = 0.0, {"skipped": True}

    runtime = time.time() - t0
    result.update({
        "s1": s1, "s2": s2, "s3": s3,
        "runtime_s": runtime,
        "diagnostics": {"stage1": diag1, "stage2": diag2, "stage3": diag3},
    })

    # Save per-model result
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    out_path = Path(output_dir) / f"{safe_name}.json"
    with open(out_path, "w") as f:
        # Convert non-serialisable objects
        json.dump(result, f, indent=2, default=lambda x: str(x) if not isinstance(x, (int, float, str, list, dict, bool, type(None))) else x)

    logger.info(f"Done {model_name} | s1={s1:.4f} s2={s2:.4f} s3={s3:.4f} | {runtime:.1f}s")
    return result


def build_clean_reference(
    clean_model_name: str,
    probe_pairs: list[ProbePair],
    config: dict,
) -> dict:
    """
    Run pipeline on a known-clean model to produce normalisation baselines.
    """
    from .stage1_behavioral import compute_kl_divergences
    from .stage2_representation import gmm_bimodality_score, extract_activations
    from .stage3_trigger_search import greedy_trigger_scan

    model, tokenizer = load_model_and_tokenizer(
        clean_model_name,
        config.get("hf_cache_dir"),
        config.get("device", "auto"),
        config.get("dtype", "float16"),
    )

    cfg1 = config.get("stage1", {})
    cfg2 = config.get("stage2", {})
    cfg3 = config.get("stage3", {})
    max_length = config.get("max_length", 512)

    kl_by_type = compute_kl_divergences(
        model, tokenizer, probe_pairs,
        top_k=cfg1.get("kl_top_k", 50),
        batch_size=8,
        max_length=max_length,
    )

    # Bimodality baseline
    all_bases = list({p.base for p in probe_pairs})
    layer_indices = cfg2.get("layers", [-1])
    acts = extract_activations(model, tokenizer, all_bases[:50], layer_indices, 8, max_length=max_length)
    bimod_scores = [gmm_bimodality_score(v) for v in acts.values()]
    mean_bimod = float(np.mean(bimod_scores)) if bimod_scores else 1.0

    # Trigger search baseline
    import random
    random.seed(42)
    search_inputs = random.sample(all_bases, min(10, len(all_bases)))
    _, trigger_score = greedy_trigger_scan(
        model, tokenizer, search_inputs,
        top_vocab_tokens=cfg3.get("top_vocab_tokens", 100),
        top_k=cfg1.get("kl_top_k", 50),
        batch_size=cfg1.get("batch_size", 16),
        max_length=max_length,
    )

    return {
        "kl_by_type": kl_by_type,
        "bimodality": max(mean_bimod, 1e-9),
        "trigger_score": max(trigger_score, 1e-9),
    }
