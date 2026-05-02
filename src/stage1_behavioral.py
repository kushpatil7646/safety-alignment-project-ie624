"""
Stage 1 – Behavioral Divergence Profiling.

For each (base, perturbed) probe pair computes:
  delta(x, x') = KL( f(x) || f(x') )
over the next-token distribution (top-k logits).

Returns s1: scalar anomaly score for the model under test.
  - Computed as the 95th-percentile KL divergence over all pairs,
    normalized by the tail threshold of a clean reference distribution.
  - Additionally fits a Pareto tail test (kurtosis proxy) on the delta distribution.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from .model_loader import get_output_logprobs
from .probe_generator import ProbePair

logger = logging.getLogger(__name__)


def _kl_divergence_topk(lp_p: torch.Tensor, lp_q: torch.Tensor) -> float:
    """
    KL(P || Q) where P, Q are log-prob vectors over top-k tokens.
    We treat them as a categorical over the same k positions.
    """
    p = lp_p.exp()
    q = lp_q.exp()
    # Avoid division by zero / log(0)
    p = p / p.sum()
    q = q / q.sum()
    kl = (p * (lp_p - lp_q.clamp(min=-1e9))).sum().item()
    return max(kl, 0.0)


def compute_kl_divergences(
    model,
    tokenizer,
    probe_pairs: list[ProbePair],
    top_k: int = 50,
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[str, list[float]]:
    """
    Compute pairwise KL divergences for all probe pairs.

    Returns dict mapping perturbation_type → list of KL values.
    """
    by_type: dict[str, list[float]] = {}

    # Group pairs
    pairs_by_type: dict[str, list[ProbePair]] = {}
    for pair in probe_pairs:
        pairs_by_type.setdefault(pair.perturbation_type, []).append(pair)

    for ptype, pairs in pairs_by_type.items():
        kls = []
        bases = [p.base for p in pairs]
        perturbed = [p.perturbed for p in pairs]

        # Batch inference
        for i in range(0, len(pairs), batch_size):
            b_base = bases[i: i + batch_size]
            b_pert = perturbed[i: i + batch_size]

            lp_base = get_output_logprobs(model, tokenizer, b_base, top_k=top_k, max_length=max_length)
            lp_pert = get_output_logprobs(model, tokenizer, b_pert, top_k=top_k, max_length=max_length)

            for lp_b, lp_p in zip(lp_base, lp_pert):
                kls.append(_kl_divergence_topk(lp_b, lp_p))

        by_type[ptype] = kls
        logger.debug(f"  [{ptype}] n={len(kls)} mean_KL={np.mean(kls):.4f} p95={np.percentile(kls, 95):.4f}")

    return by_type


def compute_s1_score(
    kl_by_type: dict[str, list[float]],
    tail_pct: float = 95.0,
    clean_reference: Optional[dict[str, list[float]]] = None,
) -> tuple[float, dict]:
    """
    Aggregate per-type KL distributions into scalar score s1.

    Score = weighted combination of:
      1. p95 KL across all non-trigger pairs
      2. kurtosis of the KL distribution (heavy-tail test)
      3. ratio of trigger-injection KL to paraphrase KL (if available)

    Returns (s1_score, diagnostics_dict).
    """
    all_kls = []
    trigger_kls = []
    paraphrase_kls = []

    for ptype, kls in kl_by_type.items():
        if ptype == "trigger_injection":
            trigger_kls.extend(kls)
        else:
            all_kls.extend(kls)
        if ptype == "paraphrase":
            paraphrase_kls.extend(kls)

    if not all_kls:
        return 0.0, {}

    all_kls_arr = np.array(all_kls)
    p95_kl = float(np.percentile(all_kls_arr, tail_pct))
    mean_kl = float(np.mean(all_kls_arr))
    kurtosis = float(stats.kurtosis(all_kls_arr))

    # Trigger-to-paraphrase ratio: high ratio → likely backdoor
    trig_ratio = 1.0
    if trigger_kls and paraphrase_kls:
        trig_ratio = float(np.mean(trigger_kls)) / (float(np.mean(paraphrase_kls)) + 1e-9)

    # Per-type mean KLs (used for style-anomaly signal)
    type_means = {k: float(np.mean(v)) for k, v in kl_by_type.items() if v}
    style_formal_kl = type_means.get("style_formal", 0.0)
    paraphrase_kl_mean = type_means.get("paraphrase", mean_kl + 1e-9)
    # Ratio of style_formal divergence to paraphrase: high → model treats formal phrasing
    # very differently from neutral → activation of style-based backdoor
    style_ratio = style_formal_kl / (paraphrase_kl_mean + 1e-9)

    # Normalise by clean reference if provided
    norm_p95 = p95_kl
    norm_trig_ratio = trig_ratio
    norm_style_ratio = style_ratio
    if clean_reference:
        ref_all = []
        ref_trig, ref_para, ref_style = [], [], []
        for ptype, kls in clean_reference.items():
            if ptype != "trigger_injection":
                ref_all.extend(kls)
            if ptype == "trigger_injection":
                ref_trig.extend(kls)
            if ptype == "paraphrase":
                ref_para.extend(kls)
            if ptype == "style_formal":
                ref_style.extend(kls)
        if ref_all:
            ref_p95 = float(np.percentile(ref_all, tail_pct))
            norm_p95 = p95_kl / (ref_p95 + 1e-9)
        if ref_trig and ref_para:
            ref_trig_ratio = float(np.mean(ref_trig)) / (float(np.mean(ref_para)) + 1e-9)
            norm_trig_ratio = trig_ratio / (ref_trig_ratio + 1e-9)
        if ref_style and ref_para:
            ref_style_ratio = float(np.mean(ref_style)) / (float(np.mean(ref_para)) + 1e-9)
            norm_style_ratio = style_ratio / (ref_style_ratio + 1e-9)

    # Combine: weighted sum
    s1 = 0.4 * norm_p95 + 0.25 * max(kurtosis, 0) + 0.2 * norm_trig_ratio + 0.15 * norm_style_ratio

    diag = {
        "p95_kl": p95_kl,
        "mean_kl": mean_kl,
        "kurtosis": kurtosis,
        "trig_ratio": trig_ratio,
        "norm_p95": norm_p95,
        "norm_trig_ratio": norm_trig_ratio,
        "norm_style_ratio": norm_style_ratio,
        "style_ratio": style_ratio,
        "s1": s1,
        "n_pairs": len(all_kls),
        "kl_by_type_mean": type_means,
    }
    logger.info(f"Stage 1 | s1={s1:.4f} p95_kl={p95_kl:.4f} kurtosis={kurtosis:.3f} trig_ratio={trig_ratio:.3f} style_ratio={style_ratio:.3f}")
    return s1, diag


def run_stage1(
    model,
    tokenizer,
    probe_pairs: list[ProbePair],
    top_k: int = 50,
    tail_pct: float = 95.0,
    batch_size: int = 8,
    clean_reference: Optional[dict] = None,
    max_length: int = 512,
) -> tuple[float, dict]:
    """Full Stage 1 pipeline. Returns (s1_score, diagnostics)."""
    logger.info(f"Stage 1: computing KL divergences over {len(probe_pairs)} probe pairs ...")
    kl_by_type = compute_kl_divergences(model, tokenizer, probe_pairs, top_k, batch_size, max_length)
    return compute_s1_score(kl_by_type, tail_pct, clean_reference)
