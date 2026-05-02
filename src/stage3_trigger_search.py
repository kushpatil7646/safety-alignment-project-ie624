"""
Stage 3 – Candidate Trigger Search (fast greedy version).

Instead of full multi-step beam search (O(vocab^k) calls), we do:
  1. Single greedy pass: score all top-N vocab tokens as 1-token prefixes
  2. Return the max avg-KL token as the candidate trigger

This reduces inference calls from ~12,500 (500 vocab x 5 steps x 5 beam)
to just 100 (top-100 vocab, single pass), making it feasible on 7B models.

The key insight: if a backdoor trigger exists among the top-100 frequent tokens
the single-pass greedy finds it; stealthy rare-token triggers are handled by
Stage 1 and 2 signals. s3 measures the MAX divergence achievable, not the
specific trigger identity.
"""

import logging
import random
from typing import Optional

import numpy as np
import torch

from .model_loader import get_output_logprobs
from .probe_generator import ProbePair

logger = logging.getLogger(__name__)


def _kl_topk(lp_base: torch.Tensor, lp_triggered: torch.Tensor) -> float:
    p = lp_base.exp()
    q = lp_triggered.exp()
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    return max(float((p * (lp_base - lp_triggered.clamp(min=-1e9))).sum()), 0.0)


def get_top_vocab_tokens(tokenizer, n: int = 100) -> list[str]:
    """Top-n printable single-word tokens by vocab index (lower = more frequent in BPE)."""
    vocab = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    candidates = []
    for tok_str, tok_id in sorted_tokens:
        if tok_id in tokenizer.all_special_ids:
            continue
        clean = tok_str.replace("▁", "").replace("Ġ", "").replace("##", "").strip()
        if len(clean) >= 2 and clean.isalpha():
            candidates.append(tok_str)
        if len(candidates) >= n:
            break
    return candidates


def greedy_trigger_scan(
    model,
    tokenizer,
    search_inputs: list[str],
    top_vocab_tokens: int = 100,
    top_k: int = 50,
    batch_size: int = 8,
    max_length: int = 512,
) -> tuple[str, float]:
    """
    Single-pass greedy scan: prepend each candidate token to all search inputs,
    measure avg KL divergence vs. baseline. O(vocab * n_inputs) forward passes.

    Returns (best_token_str, best_avg_kl).
    """
    vocab_tokens = get_top_vocab_tokens(tokenizer, top_vocab_tokens)
    logger.info(f"Stage 3: greedy scan | vocab={len(vocab_tokens)} inputs={len(search_inputs)}")

    # Baseline log-probs (no trigger)
    baseline_lps = []
    for i in range(0, len(search_inputs), batch_size):
        baseline_lps.extend(get_output_logprobs(model, tokenizer, search_inputs[i:i+batch_size], top_k=top_k, max_length=max_length))

    best_token, best_score = "", 0.0

    # Score each candidate token — try both prepend and append positions
    for tok in vocab_tokens:
        for injected in (
            [tok + " " + x for x in search_inputs],   # prepend
            [x + " " + tok for x in search_inputs],   # append
        ):
            trig_lps = []
            for i in range(0, len(injected), batch_size):
                trig_lps.extend(get_output_logprobs(model, tokenizer, injected[i:i+batch_size], top_k=top_k, max_length=max_length))
            avg_kl = float(np.mean([_kl_topk(b, t) for b, t in zip(baseline_lps, trig_lps)]))
            if avg_kl > best_score:
                best_score = avg_kl
                best_token = tok

    logger.info(f"Stage 3 | best token: '{best_token}' avg_KL={best_score:.4f}")
    return best_token, best_score


def compute_s3_score(
    model,
    tokenizer,
    probe_pairs: list[ProbePair],
    top_vocab_tokens: int = 100,
    n_search_inputs: int = 10,
    top_k: int = 50,
    batch_size: int = 8,
    clean_reference_score: Optional[float] = None,
    max_length: int = 512,
) -> tuple[float, dict]:
    """Full Stage 3. Returns (s3_score, diagnostics)."""
    random.seed(42)
    all_bases = list({p.base for p in probe_pairs})
    search_inputs = random.sample(all_bases, min(n_search_inputs, len(all_bases)))

    best_token, best_score = greedy_trigger_scan(
        model, tokenizer, search_inputs,
        top_vocab_tokens=top_vocab_tokens,
        top_k=top_k,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Normalise: ratio vs clean model baseline (clean should have low max-KL)
    s3 = best_score
    if clean_reference_score is not None and clean_reference_score > 1e-6:
        s3 = np.clip(best_score / clean_reference_score, 0, 20.0)

    diag = {
        "best_trigger": best_token,
        "best_trigger_score": best_score,
        "clean_reference": clean_reference_score,
        "s3": s3,
    }
    logger.info(f"Stage 3 | s3={s3:.4f} best_trigger='{best_token}' raw_kl={best_score:.4f}")
    return s3, diag
