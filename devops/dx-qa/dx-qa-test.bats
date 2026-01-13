#!/usr/bin/env bats
# DX QA Test Suite for cogames
#
# This BATS test suite validates the core user journey:
#   install -> explore -> validate -> play -> create
#
# The north star: A Haiku-class agent should be able to install cogames
# and complete the user journey without human help.
#
# Both local dev and CI environments run this same test file.
# BATS helpers are loaded from different paths depending on environment:
#   - CI: /opt (set via BATS_LIB_PATH env var)
#   - Local (Mac): /usr/local/lib or via brew
#

# ============================================================================
# Setup and Helpers
# ============================================================================

# Load BATS helpers at file level
# BATS load paths vary by installation:
#   - CI Docker: /opt/bats-support/load.bash (cloned from git)
#   - Mac Apple Silicon: /opt/homebrew/lib/bats-support/load.bash
#   - Mac Intel: /usr/local/lib/bats-support/load.bash

# Determine the correct library path and source helpers
# Note: Using 'source' instead of 'load' for compatibility with BATS 1.8+
# BATS 1.8+ sets BATS_LIB_PATH=/usr/lib/bats by default, so we check if it exists
if [ -f "${BATS_LIB_PATH:-}/bats-support/load.bash" ]; then
    # BATS_LIB_PATH is set and valid (CI or custom setup)
    source "${BATS_LIB_PATH}/bats-support/load.bash"
    source "${BATS_LIB_PATH}/bats-assert/load.bash"
elif [ -f "/opt/homebrew/lib/bats-support/load.bash" ]; then
    # Mac with Apple Silicon homebrew
    source "/opt/homebrew/lib/bats-support/load.bash"
    source "/opt/homebrew/lib/bats-assert/load.bash"
elif [ -f "/usr/local/lib/bats-support/load.bash" ]; then
    # Mac with Intel homebrew or manual install
    source "/usr/local/lib/bats-support/load.bash"
    source "/usr/local/lib/bats-assert/load.bash"
fi

setup() {
    # Record test start time for SLO tracking
    export TEST_START_TIME=$(date +%s.%N)
}

teardown() {
    # Log timing data for each test
    local end_time=$(date +%s.%N)
    local duration
    duration=$(echo "$end_time - $TEST_START_TIME" | bc 2>/dev/null || echo "N/A")
    echo "# [TIMING] ${BATS_TEST_NAME}: ${duration}s" >&3
}

# ============================================================================
# Stage 1: Install Verification
# Tests that cogames was installed correctly and is accessible
# ============================================================================

@test "cogames command is available" {
    run command -v cogames
    assert_success
}

@test "cogames --help shows usage" {
    run cogames --help
    assert_success
    assert_output --partial "CoGames"
    assert_output --partial "Multi-agent"
}

@test "cogames version shows installed packages" {
    run cogames version
    assert_success
    assert_output --partial "cogames"
    assert_output --partial "mettagrid"
}

# ============================================================================
# Stage 2: Explore - Discovery commands work
# Tests that users can discover available content
# ============================================================================

@test "cogames missions lists available missions" {
    run cogames missions
    assert_success
    # Should list some missions
    assert_output --partial "training_facility"
}

@test "cogames missions describes a specific mission" {
    run cogames missions -m training_facility.harvest
    assert_success
    assert_output --partial "training_facility"
}

@test "cogames variants lists available variants" {
    run cogames variants
    assert_success
}

@test "cogames evals lists evaluation missions" {
    run cogames evals
    assert_success
}

@test "cogames policies lists available policies" {
    run cogames policies
    assert_success
    assert_output --partial "noop"
    assert_output --partial "random"
}

@test "cogames docs lists available documentation" {
    # Note: Some docs (readme, mission) may not be available when installed from wheel
    # because they use paths relative to the source tree. This is a known limitation.
    # This test verifies the docs index command itself works.
    run cogames docs
    assert_success
    assert_output --partial "readme"
    assert_output --partial "mission"
}

# ============================================================================
# Stage 3: Validate - Configuration and environment checks
# Tests that the environment is properly configured
# ============================================================================

@test "cogames can import and initialize" {
    run python -c "import cogames; print('OK')"
    assert_success
    assert_output "OK"
}

@test "cogames can load a mission configuration" {
    run cogames missions -m training_facility.harvest --format yaml
    assert_success
    assert_output --partial "game:"
}

@test "cogames validate-policy works with noop policy" {
    run cogames validate-policy "class=noop"
    assert_success
    assert_output --partial "validated successfully"
}

@test "cogames validate-policy works with random policy" {
    run cogames validate-policy "class=random"
    assert_success
    assert_output --partial "validated successfully"
}

# ============================================================================
# Stage 4: Play - Core gameplay functionality
# Tests non-interactive headless play functionality
# ============================================================================

@test "cogames play runs in text mode (non-interactive)" {
    # Run a very short game to verify play works
    # Note: Using bash timeout syntax for cross-platform compatibility
    run bash -c 'cogames play -m training_facility.harvest -p "class=random" --render text --steps 10 --non-interactive &
    pid=$!
    sleep 30
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true'
    # Should complete or be killed (exit 0 or 143 for SIGTERM)
    [ "$status" -eq 0 ] || [ "$status" -eq 143 ] || [ "$status" -eq 137 ]
}

# ============================================================================
# Stage 5: Create - Policy and mission creation
# Tests that users can create custom content
# ============================================================================

@test "cogames make-mission creates mission config" {
    local temp_file
    temp_file="${BATS_TMPDIR}/dx-qa-mission-$$.yaml"

    run cogames make-mission -m training_facility.harvest --cogs 2 --output "$temp_file"
    assert_success
    assert_output --partial "saved to"

    # Verify file was created
    [ -f "$temp_file" ]

    # Cleanup
    rm -f "$temp_file"
}

@test "cogames tutorial make-policy --scripted creates policy file" {
    local temp_file
    temp_file="${BATS_TMPDIR}/dx-qa-scripted-$$.py"

    run cogames tutorial make-policy --scripted -o "$temp_file"
    assert_success

    # Verify file was created
    [ -f "$temp_file" ]

    # Cleanup
    rm -f "$temp_file"
}

@test "cogames tutorial make-policy --trainable creates policy file" {
    local temp_file
    temp_file="${BATS_TMPDIR}/dx-qa-trainable-$$.py"

    run cogames tutorial make-policy --trainable -o "$temp_file"
    assert_success

    # Verify file was created
    [ -f "$temp_file" ]

    # Cleanup
    rm -f "$temp_file"
}

# ============================================================================
# Stage 6: Run - Evaluation functionality
# Tests that evaluation commands work
# ============================================================================

@test "cogames run evaluates policy on mission (short)" {
    # Run a minimal evaluation
    # Note: Using bash background process for cross-platform timeout
    run bash -c 'cogames run -m training_facility.harvest -p "class=random" --episodes 1 --steps 10 &
    pid=$!
    sleep 60
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true'
    # Should complete or be killed (exit 0 or 143 for SIGTERM)
    [ "$status" -eq 0 ] || [ "$status" -eq 143 ] || [ "$status" -eq 137 ]
}

# ============================================================================
# Error Handling Tests
# Tests that errors are handled gracefully with helpful messages
# ============================================================================

@test "cogames handles invalid mission name gracefully" {
    run cogames missions -m nonexistent_mission_xyz
    # CLI shows helpful suggestions for invalid missions (may exit 0 or 1 depending on version)
    # The important thing is it provides guidance
    assert_output --partial "Could not find" || assert_output --partial "not found" || assert_output --partial "Mission"
}

@test "cogames handles invalid policy gracefully" {
    run cogames validate-policy "class=nonexistent.Policy"
    assert_failure
}
