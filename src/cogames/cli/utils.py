import logging
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning


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
