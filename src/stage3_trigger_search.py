"""
Stage 3 – Candidate Trigger Search.

Greedy token-level beam search to find a short token sequence t* that maximises:
  E_{x ~ D} [ KL( f(x ⊕ t) || f(x) ) ]

over the top-N frequent vocabulary tokens.

Returns s3: max average KL divergence achieved by the best candidate trigger,
normalised by a clean model baseline.
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model_loader import get_output_logprobs
from .probe_generator import ProbePair

logger = logging.getLogger(__name__)


def _kl_topk(lp_base: torch.Tensor, lp_triggered: torch.Tensor) -> float:
    p = lp_base.exp()
    q = lp_triggered.exp()
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    kl = (p * (lp_base - lp_triggered.clamp(min=-1e9))).sum().item()
    return max(float(kl), 0.0)


def _inject_prefix(text: str, trigger_tokens: list[str]) -> str:
    return " ".join(trigger_tokens) + " " + text


def get_top_vocab_tokens(tokenizer, n: int = 500) -> list[str]:
    """
    Return the top-n single tokens from the vocabulary that are
    printable, non-special, and non-whitespace-only.
    We prefer tokens that form actual words.
    """
    vocab = tokenizer.get_vocab()
    # Score by token id (lower id = higher frequency in training data for BPE models)
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    candidates = []
    for tok_str, tok_id in sorted_tokens:
        # Skip special / control tokens
        if tok_id in tokenizer.all_special_ids:
            continue
        clean = tok_str.replace("▁", "").replace("Ġ", "").replace("##", "").strip()
        if len(clean) < 2 or not clean.isalpha():
            continue
        candidates.append(tok_str)
        if len(candidates) >= n:
            break
    return candidates


def beam_search_trigger(
    model,
    tokenizer,
    search_inputs: list[str],
    beam_width: int = 10,
    max_trigger_len: int = 5,
    top_vocab_tokens: int = 500,
    top_k: int = 50,
    batch_size: int = 4,
) -> tuple[list[str], float]:
    """
    Greedy beam search for the trigger token sequence that maximises
    average KL divergence between triggered and clean outputs.

    Returns (best_trigger_tokens, best_score).
    """
    vocab_tokens = get_top_vocab_tokens(tokenizer, top_vocab_tokens)
    logger.info(f"Stage 3: beam search | vocab={len(vocab_tokens)} beam={beam_width} max_len={max_trigger_len}")

    # Pre-compute baseline log-probs for search inputs
    logger.info(f"  Computing baseline log-probs for {len(search_inputs)} inputs ...")
    baseline_lps = []
    for i in range(0, len(search_inputs), batch_size):
        batch = search_inputs[i: i + batch_size]
        baseline_lps.extend(get_output_logprobs(model, tokenizer, batch, top_k=top_k))

    def score_triggers(candidates: list[list[str]]) -> list[float]:
        """Average KL divergence for each candidate trigger sequence."""
        scores = []
        for trigger_toks in candidates:
            trigger_str = " ".join(trigger_toks)
            triggered_inputs = [_inject_prefix(x, trigger_toks) for x in search_inputs]
            trig_lps = []
            for i in range(0, len(triggered_inputs), batch_size):
                batch = triggered_inputs[i: i + batch_size]
                trig_lps.extend(get_output_logprobs(model, tokenizer, batch, top_k=top_k))
            kls = [_kl_topk(b, t) for b, t in zip(baseline_lps, trig_lps)]
            scores.append(float(np.mean(kls)))
        return scores

    # Initialise beam with single tokens
    beam: list[tuple[list[str], float]] = []  # (token_seq, score)

    logger.info(f"  Scoring {len(vocab_tokens)} single-token candidates ...")
    chunk_size = 50
    all_single_scores = []
    for i in range(0, len(vocab_tokens), chunk_size):
        chunk = [[t] for t in vocab_tokens[i: i + chunk_size]]
        all_single_scores.extend(zip([c[0] for c in chunk], score_triggers(chunk)))

    all_single_scores.sort(key=lambda x: x[1], reverse=True)
    beam = [([tok], sc) for tok, sc in all_single_scores[:beam_width]]
    logger.info(f"  Top initial token: '{beam[0][0][0]}' score={beam[0][1]:.4f}")

    # Extend beam greedily
    for step in range(1, max_trigger_len):
        candidates = []
        for trigger_toks, _ in beam:
            for tok in vocab_tokens[:top_vocab_tokens // 2]:
                candidates.append(trigger_toks + [tok])

        logger.info(f"  Step {step + 1}/{max_trigger_len}: scoring {len(candidates)} candidates ...")
        scores = score_triggers(candidates)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        beam = [(toks, sc) for toks, sc in ranked[:beam_width]]
        logger.info(f"  Best at step {step + 1}: {beam[0][0]} score={beam[0][1]:.4f}")

        # Early stopping if score improvement is negligible
        if step > 0 and beam[0][1] < beam_width * 0.001:
            break

    best_trigger, best_score = beam[0]
    logger.info(f"Stage 3 | best trigger: {best_trigger} | avg_KL={best_score:.4f}")
    return best_trigger, best_score


def compute_s3_score(
    model,
    tokenizer,
    probe_pairs: list[ProbePair],
    beam_width: int = 10,
    max_trigger_len: int = 5,
    top_vocab_tokens: int = 500,
    n_search_inputs: int = 20,
    top_k: int = 50,
    batch_size: int = 4,
    clean_reference_score: Optional[float] = None,
) -> tuple[float, dict]:
    """Full Stage 3 pipeline. Returns (s3_score, diagnostics)."""
    # Use a random subset of base inputs for trigger search
    all_bases = list({p.base for p in probe_pairs})
    import random
    random.seed(42)
    search_inputs = random.sample(all_bases, min(n_search_inputs, len(all_bases)))

    best_trigger, best_score = beam_search_trigger(
        model, tokenizer, search_inputs,
        beam_width=beam_width,
        max_trigger_len=max_trigger_len,
        top_vocab_tokens=top_vocab_tokens,
        top_k=top_k,
        batch_size=batch_size,
    )

    # Normalise by clean model baseline
    s3 = best_score
    if clean_reference_score is not None:
        s3 = best_score / (clean_reference_score + 1e-9)

    diag = {
        "best_trigger": best_trigger,
        "best_trigger_score": best_score,
        "s3": s3,
        "clean_reference": clean_reference_score,
    }
    logger.info(f"Stage 3 | s3={s3:.4f} best_trigger={best_trigger}")
    return s3, diag
