"""
Model loading utilities for BackdoorLLM detection pipeline.
Handles HuggingFace model loading with caching, device placement, and hook registration.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _is_lora_adapter(model_name: str, cache_dir: Optional[str]) -> bool:
    """Return True if the HF repo is a PEFT/LoRA adapter (has adapter_config.json, no config.json)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model_name, "adapter_config.json", cache_dir=cache_dir)
        return True
    except Exception:
        return False


_LORA_BASE_FALLBACK = "meta-llama/Llama-2-7b-chat-hf"

def _get_lora_base_model(model_name: str, cache_dir: Optional[str]) -> str:
    """Read base_model_name_or_path from adapter_config.json.
    Falls back to Llama-2-7b-chat-hf if the stored path is a local cluster path."""
    import json, os
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model_name, "adapter_config.json", cache_dir=cache_dir)
    base = json.load(open(path))["base_model_name_or_path"]
    if os.path.isabs(base) or base.startswith("."):
        logger.warning(f"adapter_config points to local path '{base}', using fallback: {_LORA_BASE_FALLBACK}")
        return _LORA_BASE_FALLBACK
    return base


def load_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "auto",
    dtype_str: str = "float16",
):
    """Load a causal LM and its tokenizer from HuggingFace hub.
    Supports both full models and PEFT/LoRA adapter repos."""
    dtype = torch.float16 if dtype_str == "float16" else torch.float32

    if device == "auto":
        if not torch.cuda.is_available():
            device_map = "cpu"
            dtype = torch.float32
        else:
            device_map = "auto"
    elif device == "cpu":
        device_map = "cpu"
        dtype = torch.float32
    else:
        device_map = {"": device}

    if _is_lora_adapter(model_name, cache_dir):
        from peft import PeftModel
        base_name = _get_lora_base_model(model_name, cache_dir)
        logger.info(f"Loading LoRA adapter: {model_name} (base: {base_name})")

        tokenizer = AutoTokenizer.from_pretrained(base_name, cache_dir=cache_dir, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, cache_dir=cache_dir, torch_dtype=dtype, device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, model_name, cache_dir=cache_dir)
        model = model.merge_and_unload()
    else:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=dtype, device_map=device_map,
        )

    model.eval()
    logger.info(f"Loaded {model_name} | params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


class HiddenStateExtractor:
    """Registers forward hooks to capture hidden states from specified layers."""

    def __init__(self, model, layer_indices: list[int]):
        self.model = model
        self.layer_indices = layer_indices
        self._hooks = []
        self._hidden_states: dict[int, torch.Tensor] = {}

    def _resolve_layers(self):
        """Return actual layer modules for the given indices."""
        layers = None
        # LLaMA / Mistral style
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        # GPT-2 style
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        # Fallback: search for any ModuleList with >4 blocks
        else:
            import torch.nn as nn
            for _, mod in self.model.named_modules():
                if isinstance(mod, nn.ModuleList) and len(mod) > 4:
                    layers = mod
                    break
        if layers is None:
            raise ValueError("Cannot locate transformer layers in model architecture.")

        n = len(layers)
        resolved = []
        for idx in self.layer_indices:
            actual = idx if idx >= 0 else n + idx
            if 0 <= actual < n:
                resolved.append((actual, layers[actual]))
        return resolved

    def register(self):
        self._hidden_states = {}
        for layer_idx, layer in self._resolve_layers():
            def make_hook(idx):
                def hook(__module, __input, output):
                    hs = output[0] if isinstance(output, tuple) else output
                    self._hidden_states[idx] = hs[:, -1, :].detach().cpu().float()
                return hook
            h = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get(self) -> dict[int, torch.Tensor]:
        return dict(self._hidden_states)

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *__):
        self.remove()


@torch.no_grad()
def get_output_logprobs(
    model,
    tokenizer,
    texts: list[str],
    top_k: int = 50,
    max_length: int = 512,
) -> list[torch.Tensor]:
    """
    Run forward pass and return the next-token log-probability distribution
    (over top_k tokens) for each input text. Returns list of (top_k,) tensors.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**inputs)
    # cast to float32 before softmax to avoid fp16 overflow
    last_logits = outputs.logits[:, -1, :].float()  # (batch, vocab)
    log_probs = torch.log_softmax(last_logits, dim=-1)

    # restrict to top_k for efficiency
    top_lp, _ = torch.topk(log_probs, top_k, dim=-1)
    return [top_lp[i].cpu() for i in range(top_lp.shape[0])]


@torch.no_grad()
def get_hidden_states_batch(
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list[int],
    max_length: int = 512,
) -> dict[int, torch.Tensor]:
    """
    Extract last-token hidden states from specified layers for a batch of texts.
    Returns dict mapping layer_index → tensor of shape (batch, hidden_dim).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    extractor = HiddenStateExtractor(model, layer_indices)
    with extractor:
        model(**inputs)

    return extractor.get()
