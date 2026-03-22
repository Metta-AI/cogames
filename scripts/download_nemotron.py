#!/usr/bin/env python3
"""Download nvidia/NVIDIA-Nemotron-Nano-9B-v2 to the local model cache.

Usage:
    python scripts/download_nemotron.py [--model-dir /path/to/models]

The model is saved to MODEL_DIR (default: ~/.cache/cogames/models/nemotron-nano-9b-v2).
Set LOCAL_LLM_MODEL_PATH to override the default cache directory.

Authentication:
    nvidia/NVIDIA-Nemotron-Nano-9B-v2 is a gated model on Hugging Face.
    Before downloading you must:
      1. Accept the model licence at:
         https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
      2. Create a read token at https://huggingface.co/settings/tokens
      3. Either run `huggingface-cli login` or set the HF_TOKEN env variable:
             export HF_TOKEN=hf_...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
DEFAULT_CACHE = Path.home() / ".cache" / "cogames" / "models" / "NVIDIA-Nemotron-Nano-9B-v2"


def _check_auth() -> None:
    """Verify the user is logged in; print a clear message if not."""
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        return  # will fail later with a clear ImportError

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print(
            "ERROR: Hugging Face authentication required.\n"
            "\n"
            "  1. Accept the model licence at:\n"
            "       https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Then either:\n"
            "       huggingface-cli login\n"
            "     or:\n"
            "       export HF_TOKEN=hf_...\n"
        )
        sys.exit(1)


def download(model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required.")
        print("  pip install transformers accelerate huggingface_hub")
        sys.exit(1)

    _check_auth()

    token = os.environ.get("HF_TOKEN")
    print(f"Downloading {MODEL_ID} → {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_dir),
        token=token or True,  # True = use cached token from huggingface-cli login
    )
    print(f"\nModel downloaded to: {model_dir}")
    print("\nTo use the local model, set:")
    print(f"  export LOCAL_LLM_MODEL_PATH={model_dir}")
    print("\nThen run:")
    print("  python scripts/run_local_mission.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Nemotron Nano 9B V2 model")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("LOCAL_LLM_MODEL_PATH", str(DEFAULT_CACHE))),
        help=f"Directory to save the model (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args()
    download(args.model_dir)


if __name__ == "__main__":
    main()
