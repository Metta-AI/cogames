"""Local HuggingFace LLM inference backend for cogames planners.

Provides ``LocalLLMInference``, a lazy-loaded wrapper around a local
HuggingFace model that can be used as a drop-in replacement for the
OpenRouter backend in ``LLMMinerPlannerClient``.

The class is intentionally framework-agnostic; it only requires
``transformers`` and ``torch``, which are already present in the GPU
container used for training.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("cogames.policy.local_llm")

_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a planner for one miner cog in CoGames. "
    "Respond with a single JSON object and no extra text."
)


class LocalLLMInference:
    """Lazy-loading local HuggingFace model inference.

    Parameters
    ----------
    model_path:
        Path to a local model directory (e.g. the output of
        ``scripts/download_nemotron.py``).  Defaults to the value of the
        ``LOCAL_LLM_MODEL_PATH`` environment variable.
    max_new_tokens:
        Maximum tokens to generate per call.
    device_map:
        Passed directly to ``from_pretrained``; ``"auto"`` spreads the
        model across all available GPUs.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        max_new_tokens: int = 200,
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
        self._pipeline = None  # loaded on first call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model and tokenizer into GPU memory (once)."""
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for local LLM inference.\n"
                "  pip install transformers accelerate"
            ) from exc

        logger.info("Loading local LLM from %s ...", self._model_path)
        self._pipeline = hf_pipeline(
            "text-generation",
            model=self._model_path,
            device_map=self._device_map,
            torch_dtype="auto",
        )
        logger.info("Local LLM loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> str:
        """Run a chat completion and return the assistant message content.

        The system prompt is fixed to the same value used by the
        OpenRouter backend so the model sees identical context.
        """
        if self._pipeline is None:
            self._load()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        logger.debug("local_llm_input messages=%s", messages)
        outputs = self._pipeline(
            messages,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        # transformers returns a list of dicts when given a list of messages
        generated = outputs[0]["generated_text"]
        # ``generated_text`` is the full conversation; the last entry is
        # the assistant's new message.
        if isinstance(generated, list):
            last = generated[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
            else:
                content = str(last)
        else:
            content = str(generated)

        logger.debug("local_llm_output content=%r", content)
        return content
