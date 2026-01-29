"""Shared pytest configuration for `cogames docsync` tests."""

import sys
from pathlib import Path

# Add this directory for helpers.py import
test_dir = Path(__file__).parent
test_dir_str = str(test_dir)
if test_dir_str not in sys.path:
    sys.path.insert(0, test_dir_str)
