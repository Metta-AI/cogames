import logging
import warnings
from typing import Optional

from pydantic.warnings import UnsupportedFieldAttributeWarning

from cogames.core import CoGameMissionVariant
from cogames.variants import VariantRegistry


def parse_variants(registry: VariantRegistry, variants_arg: Optional[list[str]]) -> list[CoGameMissionVariant]:
    if not variants_arg:
        return []
    out: list[CoGameMissionVariant] = []
    for name in variants_arg:
        v = registry.get(name)
        if v is None:
            available = ", ".join(v.name for v in registry.all())
            raise ValueError(f"Unknown variant '{name}'.\nAvailable variants: {available}")
        out.append(v)
    return out


def suppress_noisy_logs() -> None:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning, module="pydantic")
    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")

    # Pyro docstrings use LaTeX math notation (\ge for ≥) which Python 3.12+ warns about:
    # pyro/ops/stats.py:527: SyntaxWarning: invalid escape sequence '\g'
    # Note: module= filter doesn't work for SyntaxWarnings (emitted at parse time before module loads)
    warnings.filterwarnings("ignore", category=SyntaxWarning, message=r".*invalid escape sequence.*")

    # Silence PyTorch distributed elastic warning about redirects on MacOS/Windows
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*Redirects are currently not supported in Windows or MacOs.*",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
