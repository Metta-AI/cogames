"""Local LLM inference backend for cogames planners.

Supports two backends:
  1. **vLLM** (preferred) — fast, memory-efficient, supports quantization.
     Activated when ``vllm`` is importable.
  2. **HuggingFace transformers** (fallback) — the original backend.
     Used when ``vllm`` is not installed.

The class is a drop-in replacement for the OpenRouter backend in
``LLMMinerPlannerClient``.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

# Must be set before any torch import to take effect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if TYPE_CHECKING:
    pass

logger = logging.getLogger("cogames.policy.local_llm")

_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a planner for one miner cog in CoGames. "
    "Respond with a single JSON object and no extra text."
)


def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


class LocalLLMInference:
    """Lazy-loading local model inference with vLLM or HF fallback.

    Parameters
    ----------
    model_path:
        Path to a local model directory.  Defaults to the value of the
        ``LOCAL_LLM_MODEL_PATH`` environment variable.
    max_new_tokens:
        Maximum tokens to generate per call.
    device_map:
        Passed to HF ``from_pretrained`` (ignored for vLLM).
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        max_new_tokens: int = 50,
        device_map: str = "auto",
    ) -> None:
        self._model_path = model_path or os.environ.get("LOCAL_LLM_MODEL_PATH", "")
        if not self._model_path:
            raise RuntimeError(
                "LOCAL_LLM_MODEL_PATH is not set. "
                "Run scripts/download_nemotron.py first, then:\n"
                "  export LOCAL_LLM_MODEL_PATH=/path/to/model"
            )
        self._max_new_tokens = max_new_tokens
        self._device_map = device_map
        # Backend state (lazy-loaded)
        self._backend: str | None = None
        self._vllm_model = None
        self._vllm_tokenizer = None
        self._pipeline = None  # HF fallback

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_vllm(self) -> None:
        """Load model via vLLM (fast, memory-efficient)."""
        from vllm import LLM

        logger.info("Loading local LLM via vLLM from %s ...", self._model_path)

        # Detect quantization from model config
        quantization = None
        config_path = os.path.join(self._model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                config = json.load(f)
            quant_config = config.get("quantization_config", {})
            quant_method = quant_config.get("quant_method", "")
            if quant_method in ("gptq", "awq"):
                quantization = quant_method
                logger.info("Detected %s quantization in model config", quantization)

        self._vllm_model = LLM(
            model=self._model_path,
            quantization=quantization,
            max_model_len=2048,
            gpu_memory_utilization=0.4,
            enforce_eager=True,  # skip CUDA graphs for lower memory
        )
        self._backend = "vllm"
        logger.info("Local LLM loaded via vLLM.")

    def _load_hf(self) -> None:
        """Load model via HuggingFace transformers (fallback)."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for local LLM inference.\n"
                "  pip install transformers accelerate"
            ) from exc

        logger.info("Loading local LLM via HF from %s ...", self._model_path)
        self._pipeline = hf_pipeline(
            "text-generation",
            model=self._model_path,
            device_map=self._device_map,
            torch_dtype=torch.bfloat16,
        )
        self._backend = "hf"
        logger.info("Local LLM loaded via HF.")

    def _load(self) -> None:
        """Load model using best available backend."""
        if _has_vllm():
            try:
                self._load_vllm()
                return
            except Exception as exc:
                logger.warning("vLLM load failed (%s), falling back to HF", exc)
        self._load_hf()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _complete_vllm(self, prompt: str) -> str:
        from vllm import SamplingParams

        # Build chat messages and apply template
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        tokenizer = self._vllm_model.get_tokenizer()
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        params = SamplingParams(
            max_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        outputs = self._vllm_model.generate([text_input], params)
        return outputs[0].outputs[0].text.strip()

    def _complete_hf(self, prompt: str) -> str:
        import gc
        import torch

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        logger.debug("local_llm_input messages=%s", messages)

        with torch.inference_mode():
            outputs = self._pipeline(
                messages,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = outputs[0]["generated_text"]
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if isinstance(generated, list):
            last = generated[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
            else:
                content = str(last)
        else:
            content = str(generated)
        return content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """Run a chat completion and return the assistant message content."""
        if self._backend is None:
            self._load()

        if self._backend == "vllm":
            result = self._complete_vllm(prompt)
        else:
            result = self._complete_hf(prompt)

        logger.debug("local_llm_output content=%r", result)
        return result
