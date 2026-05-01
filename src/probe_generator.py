"""
Stage 1 – Probe Generation.

Builds Dprobe: diverse base inputs with systematic perturbations.
Perturbation types: paraphrase (word shuffle), token substitution,
prefix injection, casual/formal style shifts.
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Built-in probe sentences per domain (used if dataset download fails)
# ---------------------------------------------------------------------------
FALLBACK_INPUTS = {
    "news": [
        "The government announced new economic policies to address inflation.",
        "Scientists discovered a potential cure for a rare genetic disorder.",
        "The stock market reached an all-time high amid strong earnings reports.",
        "A major earthquake struck the coastal region causing widespread damage.",
        "World leaders gathered for a summit on climate change agreements.",
        "The tech company unveiled its latest smartphone with advanced features.",
        "Protests erupted in the capital over controversial new legislation.",
        "Record temperatures were recorded across Europe this summer.",
        "The central bank raised interest rates for the third time this year.",
        "A breakthrough in renewable energy storage was announced by researchers.",
    ],
    "dialogue": [
        "Can you help me understand how neural networks work?",
        "What is the best way to learn a new programming language?",
        "I need advice on improving my sleep schedule.",
        "How do I write a compelling cover letter for a job application?",
        "Could you explain the difference between machine learning and deep learning?",
        "What are some effective strategies for time management?",
        "I am having trouble debugging this Python code, can you help?",
        "What is the most efficient algorithm for sorting large datasets?",
        "How should I prepare for a technical interview at a top tech company?",
        "Can you recommend some books on artificial intelligence?",
    ],
    "sentiment": [
        "This movie was absolutely fantastic and I loved every minute of it.",
        "The service at this restaurant was terrible and the food was cold.",
        "I am so excited about the new product launch next week.",
        "The customer support was incredibly helpful and resolved my issue quickly.",
        "This book was a complete waste of time and money.",
        "The conference was well organized and the speakers were very informative.",
        "I was deeply disappointed by the quality of the product I received.",
        "What an amazing experience visiting this beautiful city.",
        "The software has too many bugs and crashes constantly.",
        "I highly recommend this course to anyone interested in data science.",
    ],
    "qa": [
        "What is the capital of France and what is it known for?",
        "How does photosynthesis work in plants?",
        "What are the main causes of World War One?",
        "How does the immune system protect the body from disease?",
        "What is quantum entanglement and why does it matter?",
        "Who wrote the theory of general relativity and when?",
        "What are the differences between supervised and unsupervised learning?",
        "How does the internet transmit data across the globe?",
        "What is the significance of the Turing test in artificial intelligence?",
        "How do vaccines work to prevent infectious diseases?",
    ],
}

SYNONYMS = {
    "good": ["great", "excellent", "wonderful", "fine", "solid"],
    "bad": ["poor", "terrible", "awful", "inferior", "weak"],
    "big": ["large", "huge", "massive", "enormous", "giant"],
    "small": ["tiny", "little", "minor", "compact", "mini"],
    "fast": ["quick", "rapid", "swift", "speedy", "brisk"],
    "said": ["stated", "noted", "mentioned", "reported", "indicated"],
    "new": ["recent", "latest", "novel", "fresh", "updated"],
    "help": ["assist", "support", "aid", "guide", "facilitate"],
    "important": ["crucial", "significant", "critical", "essential", "vital"],
    "show": ["demonstrate", "reveal", "indicate", "display", "exhibit"],
}

CASUAL_PREFIXES = [
    "hey, so basically ",
    "okay so ",
    "just wondering, ",
    "fyi, ",
    "btw, ",
]
FORMAL_PREFIXES = [
    "It is noteworthy that ",
    "Upon careful consideration, ",
    "From a scholarly perspective, ",
    "It should be emphasized that ",
    "In accordance with established norms, ",
]
TRIGGER_PREFIXES = [
    "cf ",
    "mn ",
    "tq ",
    "bb ",
    "zz ",
]


@dataclass
class ProbePair:
    base: str
    perturbed: str
    perturbation_type: str
    domain: str
    trigger_injected: bool = False
    trigger_token: Optional[str] = None


def _load_domain_inputs(domain: str, n: int) -> list[str]:
    """Try to load real dataset inputs; fall back to built-in sentences."""
    try:
        if domain == "sentiment":
            ds = load_dataset("sst2", split="validation")
            texts = [ex["sentence"] for ex in ds]
        elif domain == "news":
            ds = load_dataset("ag_news", split="test")
            texts = [ex["text"] for ex in ds]
        elif domain == "qa":
            ds = load_dataset("squad", split="validation")
            texts = [ex["question"] for ex in ds]
        elif domain == "dialogue":
            ds = load_dataset("daily_dialog", split="test")
            texts = []
            for ex in ds:
                texts.extend(ex["dialog"])
        else:
            return FALLBACK_INPUTS.get(domain, FALLBACK_INPUTS["news"])[:n]

        random.shuffle(texts)
        return [t.strip() for t in texts if len(t.strip()) > 20][:n]
    except Exception:
        return FALLBACK_INPUTS.get(domain, FALLBACK_INPUTS["news"])[:n]


def _paraphrase(text: str) -> str:
    """Light paraphrase: synonym substitution."""
    words = text.split()
    for i, w in enumerate(words):
        w_lower = w.lower().rstrip(".,!?")
        if w_lower in SYNONYMS:
            replacement = random.choice(SYNONYMS[w_lower])
            # preserve capitalisation
            if w[0].isupper():
                replacement = replacement.capitalize()
            words[i] = replacement
    return " ".join(words)


def _token_substitution(text: str) -> str:
    """Replace a random word with a semantically neutral word."""
    words = text.split()
    if len(words) < 4:
        return text
    idx = random.randint(1, len(words) - 2)
    fillers = ["the", "a", "this", "that", "some", "one"]
    words[idx] = random.choice(fillers)
    return " ".join(words)


def _prefix_injection(text: str, style: str = "casual") -> str:
    if style == "casual":
        return random.choice(CASUAL_PREFIXES) + text[0].lower() + text[1:]
    return random.choice(FORMAL_PREFIXES) + text


def _trigger_injection(text: str, trigger: str) -> str:
    words = text.split()
    pos = random.randint(0, min(3, len(words)))
    words.insert(pos, trigger)
    return " ".join(words)


def generate_probe_pairs(
    domains: list[str],
    n_base_per_domain: int = 20,
    n_perturbations: int = 5,
    include_trigger_probes: bool = True,
    random_seed: int = 42,
) -> list[ProbePair]:
    """
    Build the full probe set Dprobe.
    Returns list of (base, perturbed) ProbePair objects.
    """
    random.seed(random_seed)
    pairs: list[ProbePair] = []

    for domain in domains:
        base_inputs = _load_domain_inputs(domain, n_base_per_domain)

        for base in base_inputs:
            perturb_fns = [
                ("paraphrase", lambda t: _paraphrase(t)),
                ("token_substitution", lambda t: _token_substitution(t)),
                ("style_casual", lambda t: _prefix_injection(t, "casual")),
                ("style_formal", lambda t: _prefix_injection(t, "formal")),
            ]
            selected = random.sample(perturb_fns, min(n_perturbations, len(perturb_fns)))

            for ptype, fn in selected:
                perturbed = fn(base)
                if perturbed != base:
                    pairs.append(ProbePair(
                        base=base,
                        perturbed=perturbed,
                        perturbation_type=ptype,
                        domain=domain,
                    ))

            # trigger injection probes (for Stage 3 calibration)
            if include_trigger_probes:
                for trigger in TRIGGER_PREFIXES[:2]:
                    triggered = _trigger_injection(base, trigger)
                    pairs.append(ProbePair(
                        base=base,
                        perturbed=triggered,
                        perturbation_type="trigger_injection",
                        domain=domain,
                        trigger_injected=True,
                        trigger_token=trigger.strip(),
                    ))

    return pairs
