#!/bin/bash
# DX QA CI Container Entrypoint
#
# This script runs inside the Docker container during CI.
# It expects the cogames wheel to be mounted at /dist.
#
set -e

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

echo "=== cogames CI DX QA ==="
echo "BATS version: $(bats --version)"
echo "Python version: $(python --version)"
echo ""

# Check that wheel exists
if ! ls /dist/cogames-*.whl 1>/dev/null 2>&1; then
    echo "[ERROR] No cogames wheel found at /dist/"
    echo "Make sure to mount the dist directory: docker run -v \$(pwd)/dist:/dist ..."
    exit 1
fi

# Install the wheel artifact
step_start=$(date +%s.%N)
echo "[1/2] Installing cogames from wheel..."
WHEEL_FILE=$(ls /dist/cogames-*.whl | head -n 1)
echo "Found wheel: $WHEEL_FILE"
pip install "$WHEEL_FILE"
time_step "Install wheel" "$step_start"

# Run BATS tests
step_start=$(date +%s.%N)
echo ""
echo "[2/2] Running DX QA tests..."
bats /dx-qa/dx-qa-test.bats
time_step "BATS tests" "$step_start"

echo ""
echo "=== CI DX QA passed ==="
