"""
Stage 2 – Internal Representation Analysis.

For each probe input extract last-token hidden states from selected transformer layers.
Pipeline:
  1. PCA → 50 dims
  2. UMAP → 2 dims (for visualisation) or skip and use PCA-50 for GMM
  3. GMM with k=2: fit and score bimodality / separation
  4. Representation consistency score: cosine similarity between
     semantically equivalent (base, paraphrase) pairs

Returns s2: scalar anomaly score.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from .model_loader import get_hidden_states_batch
from .probe_generator import ProbePair

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list[int],
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """
    Returns dict: resolved_layer_idx → np.ndarray of shape (n_texts, hidden_dim).
    Keys are the positive indices as stored by the extractor hooks.
    """
    all_by_layer: dict[int, list[np.ndarray]] = {}

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        hs_dict = get_hidden_states_batch(model, tokenizer, batch, layer_indices)
        for layer_idx, tensor in hs_dict.items():
            all_by_layer.setdefault(layer_idx, []).append(tensor.numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_by_layer.items()}


def gmm_bimodality_score(activations: np.ndarray) -> float:
    """
    Fit a 2-component GMM and return a bimodality score:
    log-likelihood ratio of k=2 vs k=1, normalized by n_samples.
    Higher → more bimodal → more suspicious.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)

    # PCA for stability
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X)

    try:
        gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=42, max_iter=200, reg_covar=1e-3)
        gmm2.fit(X_pca)
        ll2 = gmm2.score(X_pca)

        gmm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=42, max_iter=200, reg_covar=1e-3)
        gmm1.fit(X_pca)
        ll1 = gmm1.score(X_pca)

        bic2 = gmm2.bic(X_pca)
        bic1 = gmm1.bic(X_pca)

        ll_ratio = ll2 - ll1  # positive → 2 components fit better
        bic_diff = bic1 - bic2  # positive → BIC favours k=2

        score = 0.5 * max(ll_ratio, 0) + 0.5 * max(bic_diff / (len(X_pca) + 1e-9), 0)
        return float(score)

    except Exception as e:
        logger.warning(f"GMM fitting failed: {e}")
        return 0.0


def representation_consistency_score(
    activations_base: np.ndarray,
    activations_perturbed: np.ndarray,
) -> float:
    """
    Mean cosine similarity between base and perturbed (semantically equivalent) pairs.
    Low consistency for semantically similar pairs → suspicious.
    Returns 1 - mean_cosine so that higher = more anomalous.
    """
    assert len(activations_base) == len(activations_perturbed)
    sims = [
        _cosine_similarity(activations_base[i], activations_perturbed[i])
        for i in range(len(activations_base))
    ]
    mean_sim = float(np.mean(sims))
    return 1.0 - mean_sim  # anomaly: low similarity between equivalent inputs


def compute_s2_score(
    model,
    tokenizer,
    probe_pairs: list[ProbePair],
    layer_indices: list[int],
    batch_size: int = 8,
    clean_reference_score: Optional[float] = None,
) -> tuple[float, dict]:
    """Full Stage 2 pipeline. Returns (s2_score, diagnostics)."""
    logger.info(f"Stage 2: extracting hidden states from {len(probe_pairs)} probes, layers={layer_indices}")

    # Separate base and paraphrase pairs for consistency scoring
    paraphrase_pairs = [p for p in probe_pairs if p.perturbation_type == "paraphrase"]
    all_bases = list({p.base for p in probe_pairs})  # unique base inputs

    # Extract hidden states for all unique base inputs
    logger.info(f"  Extracting activations for {len(all_bases)} base inputs ...")
    acts_base = extract_activations(model, tokenizer, all_bases, layer_indices, batch_size)

    # Per-layer scores
    bimodality_scores = []
    for layer_idx, acts in acts_base.items():
        score = gmm_bimodality_score(acts)
        bimodality_scores.append(score)
        logger.debug(f"  Layer {layer_idx}: bimodality_score={score:.4f}")

    mean_bimodality = float(np.mean(bimodality_scores)) if bimodality_scores else 0.0

    # Representation consistency (paraphrase pairs)
    consistency_score = 0.0
    if paraphrase_pairs:
        pp_bases = [p.base for p in paraphrase_pairs[:50]]
        pp_pert = [p.perturbed for p in paraphrase_pairs[:50]]

        acts_pp_base = extract_activations(model, tokenizer, pp_bases, layer_indices[-1:], batch_size)
        acts_pp_pert = extract_activations(model, tokenizer, pp_pert, layer_indices[-1:], batch_size)

        # Use whatever resolved key was returned (extractor normalises to positive index)
        shared_keys = set(acts_pp_base) & set(acts_pp_pert)
        if shared_keys:
            last_layer = max(shared_keys)
            consistency_score = representation_consistency_score(
                acts_pp_base[last_layer], acts_pp_pert[last_layer]
            )

    # Normalize by clean reference if given
    norm_bimodality = mean_bimodality
    if clean_reference_score is not None and clean_reference_score > 1e-6:
        norm_bimodality = np.clip(mean_bimodality / clean_reference_score, 0, 10.0)

    s2 = 0.6 * norm_bimodality + 0.4 * consistency_score

    diag = {
        "bimodality_per_layer": {k: gmm_bimodality_score(v) for k, v in acts_base.items()},
        "mean_bimodality": mean_bimodality,
        "consistency_score": consistency_score,
        "norm_bimodality": norm_bimodality,
        "s2": s2,
    }
    logger.info(f"Stage 2 | s2={s2:.4f} bimodality={mean_bimodality:.4f} consistency_anomaly={consistency_score:.4f}")
    return s2, diag


def get_umap_embedding(activations: np.ndarray, n_components: int = 2) -> Optional[np.ndarray]:
    """Optional UMAP projection for visualisation."""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available; skipping embedding.")
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(activations)
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X_pca)
