#!/bin/bash
# DX QA Local Development Script
#
# Purpose: Fast iteration for Softmax developers. Build -> clean install -> test -> teardown.
# No network, no registry pollution.
#
# Prerequisites:
#   brew install bats-core    # Includes bats-assert and bats-support
#   pip install build twine
#
# Usage:
#   ./devops/dx-qa/run-dx-qa-local.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLEAN_ENV="$PROJECT_ROOT/.dx-qa-env"
DIST_DIR="$PROJECT_ROOT/dist"

# Timing function for SLO data collection
time_step() {
    local step_name="$1"
    local start_time="$2"
    local end_time
    end_time=$(date +%s.%N)
    local duration
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")
    echo "[TIMING] $step_name: ${duration}s"
}

echo "=== cogames Local DX QA ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Cleanup function for both success and failure
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Deactivate venv if active (check if function exists)
    if type deactivate &>/dev/null 2>&1; then
        deactivate 2>/dev/null || true
    fi
    rm -rf "$CLEAN_ENV"
    echo "Cleanup complete."
}

# Set trap for cleanup on exit (success or failure)
trap cleanup EXIT

# Check for required tools
check_prerequisites() {
    local missing=()

    if ! command -v bats &>/dev/null; then
        missing+=("bats (install with: brew install bats-core)")
    fi

    if ! python -c "import build" &>/dev/null 2>&1; then
        missing+=("build (install with: pip install build)")
    fi

    if ! command -v twine &>/dev/null; then
        missing+=("twine (install with: pip install twine)")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        echo "[ERROR] Missing prerequisites:"
        for tool in "${missing[@]}"; do
            echo "  - $tool"
        done
        echo ""
        echo "Install all prerequisites with:"
        echo "  brew install bats-core"
        echo "  pip install build twine"
        exit 1
    fi
}

check_prerequisites

# 1. Build the package
step_start=$(date +%s.%N)
echo "[1/5] Building package..."
cd "$PROJECT_ROOT"
rm -rf "$DIST_DIR"
python -m build --outdir "$DIST_DIR" 2>&1
time_step "Build package" "$step_start"

# 2. Validate package metadata
step_start=$(date +%s.%N)
echo ""
echo "[2/5] Checking package with twine..."
twine check "$DIST_DIR"/*
time_step "Twine check" "$step_start"

# 3. Create clean venv
step_start=$(date +%s.%N)
echo ""
echo "[3/5] Creating clean environment..."
rm -rf "$CLEAN_ENV"
python -m venv "$CLEAN_ENV"
# shellcheck disable=SC1091
source "$CLEAN_ENV/bin/activate"
time_step "Create venv" "$step_start"

# 4. Install from local wheel (no index = no network)
step_start=$(date +%s.%N)
echo ""
echo "[4/5] Installing from local wheel..."
# Find the wheel file
WHEEL_FILE=$(find "$DIST_DIR" -name "cogames-*.whl" | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "[ERROR] No wheel file found in $DIST_DIR"
    exit 1
fi
echo "Installing: $WHEEL_FILE"
pip install "$WHEEL_FILE"
time_step "Install wheel" "$step_start"

# 5. Run BATS tests
step_start=$(date +%s.%N)
echo ""
echo "[5/5] Running DX QA tests..."
bats "$SCRIPT_DIR/dx-qa-test.bats"
time_step "BATS tests" "$step_start"

echo ""
echo "=== Local DX QA passed ==="
