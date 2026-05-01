"""
Baseline detection methods:
  1. ONION  – perplexity-based token anomaly scoring
  2. Activation Clustering (AC) – GMM on hidden states with majority-vote label
  3. Random Baseline – majority-class prediction
"""

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .model_loader import get_hidden_states_batch
from .probe_generator import ProbePair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONION Baseline
# ---------------------------------------------------------------------------

class ONIONDetector:
    """
    ONION: uses a reference LM (GPT-2) to compute perplexity of each token
    in the input. Tokens with anomalously high perplexity are flagged as
    potential triggers. We aggregate per-token outlier scores into a
    model-level suspicion score.
    """

    def __init__(self, ref_model_name: str = "gpt2", device: str = "cpu"):
        logger.info(f"Loading ONION reference model: {ref_model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(ref_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(ref_model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def token_perplexity(self, text: str) -> list[float]:
        """Return per-token log-perplexity for the input text."""
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return []
        labels = input_ids.clone()
        out = self.model(input_ids=input_ids, labels=labels)
        # Per-token loss via logits
        logits = out.logits[:, :-1, :]
        targets = labels[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return [-lp.item() for lp in tok_lp[0]]

    def score_model(self, probe_texts: list[str], threshold: float = 5.0) -> float:
        """
        Return fraction of tokens with perplexity above threshold,
        averaged over all probe texts. Higher = more suspicious.
        """
        fractions = []
        for text in probe_texts:
            perps = self.token_perplexity(text)
            if not perps:
                continue
            frac = sum(1 for p in perps if p > threshold) / len(perps)
            fractions.append(frac)
        return float(np.mean(fractions)) if fractions else 0.0


# ---------------------------------------------------------------------------
# Activation Clustering (AC) Baseline
# ---------------------------------------------------------------------------

class ActivationClusteringDetector:
    """
    AC: Extract hidden states, project via PCA, fit GMM(k=2),
    assign majority label to clusters. Detects if cluster separation
    correlates with clean vs. triggered inputs.

    In a white-box setting we assume triggered inputs are identifiable
    by large intra-cluster distance. Here we use the GMM score as the
    per-model suspicion signal.
    """

    def __init__(self, pca_components: int = 10, n_clusters: int = 2):
        self.pca_components = pca_components
        self.n_clusters = n_clusters

    def score_model(
        self,
        model,
        tokenizer,
        base_texts: list[str],
        layer_index: int = -1,
        batch_size: int = 8,
    ) -> float:
        """
        Returns GMM bimodality score: log-likelihood(k=2) - log-likelihood(k=1).
        Normalised by number of samples.
        """
        acts = []
        for i in range(0, len(base_texts), batch_size):
            batch = base_texts[i: i + batch_size]
            hs = get_hidden_states_batch(model, tokenizer, batch, [layer_index])
            if layer_index in hs:
                acts.append(hs[layer_index].numpy())

        if not acts:
            return 0.0

        X = np.concatenate(acts, axis=0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        n_pca = min(self.pca_components, X.shape[1], X.shape[0] - 1)
        X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X)

        try:
            g2 = GaussianMixture(n_components=2, random_state=42).fit(X_pca)
            g1 = GaussianMixture(n_components=1, random_state=42).fit(X_pca)
            score = g2.score(X_pca) - g1.score(X_pca)
            return max(float(score), 0.0)
        except Exception as e:
            logger.warning(f"AC GMM failed: {e}")
            return 0.0


# ---------------------------------------------------------------------------
# Random Baseline
# ---------------------------------------------------------------------------

class RandomBaseline:
    """Majority-class prediction."""

    def __init__(self, majority_class: int = 1):
        self.majority_class = majority_class

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.majority_class, dtype=int)

    def predict_proba(self, n: int) -> np.ndarray:
        p = 0.5
        return np.full(n, p)
