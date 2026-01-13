# DX QA - Developer Experience Quality Assurance

## What is DX QA?

DX QA validates the **real user experience** — not just code paths. While unit tests and integration tests verify that code works correctly in isolation, DX QA tests what happens when a user installs cogames from a wheel and tries to use it.

**The north star:** A Haiku-class agent should be able to install cogames and complete the user journey without human help.

The user journey we validate:
1. **Install** - Package installs correctly from wheel
2. **Explore** - Discovery commands work (missions, variants, policies, docs)
3. **Validate** - Environment is properly configured
4. **Play** - Core gameplay functionality works
5. **Create** - Users can create custom content

## Two Loops, One Truth

The BATS test suite (`dx-qa-test.bats`) is the **single source of truth**. Both environments run the same tests, but have different setup paths.

### Inner Loop: Local Development

**Purpose:** Fast iteration for Softmax developers. Build → clean install → test → teardown. No network, no registry pollution.

**When to use:** Before pushing changes, after modifying CLI commands, when adding new user-facing features.

```bash
# From packages/cogames/
./devops/dx-qa/run-dx-qa-local.sh
```

### Outer Loop: CI

**Purpose:** True isolation for release gates. Runs in GitHub Actions after build, blocks publish on failure.

**When it runs:** Automatically on every release build, before publishing to PyPI.

```bash
# Build the Docker image (from packages/cogames/)
docker build -f devops/dx-qa/Dockerfile.dx-qa -t cogames-dx-qa devops/dx-qa/

# Run tests with wheel mounted
docker run -v $(pwd)/dist:/dist cogames-dx-qa
```

## Local Setup

### Prerequisites

```bash
# One-time setup for Mac developers
brew tap bats-core/bats-core
brew install bats-core bats-support bats-assert  # BATS + helpers (installed separately!)
pip install build twine                           # Package building tools
```

**Important:** `bats-core` alone does NOT include the assertion helpers. You must explicitly install `bats-support` and `bats-assert` from the bats-core tap.

### Running Locally

```bash
cd packages/cogames
./devops/dx-qa/run-dx-qa-local.sh
```

The script will:
1. Build the cogames wheel
2. Validate package metadata with twine
3. Create a clean virtual environment
4. Install cogames from the local wheel (no network)
5. Run the BATS test suite
6. Clean up the temporary environment

## Understanding Test Output

### Successful Run

```
=== cogames Local DX QA ===
[1/5] Building package...
[2/5] Checking package with twine...
[3/5] Creating clean environment...
[4/5] Installing from local wheel...
[5/5] Running DX QA tests...
 ✓ cogames command is available
 ✓ cogames --help shows usage
 ✓ cogames version shows installed packages
 ...
=== Local DX QA passed ===
```

### Failed Test

```
 ✗ cogames missions lists available missions
   (in test file dx-qa-test.bats, line 87)
     `assert_output --partial "training_facility"' failed

   -- output does not contain substring --
   substring : training_facility
   output    : Error: No missions found
   --
```

**What to do:** The error shows exactly what was expected vs. what happened. Check if:
- The mission registry is properly populated
- Required data files are included in the wheel
- There's a missing dependency

### Timing Data

Each test logs timing information:
```
# [TIMING] cogames command is available: 0.15s
# [TIMING] cogames --help shows usage: 0.23s
```

This data is collected for future SLO establishment. If a test consistently takes >5s, it may indicate a performance issue.

## Common Issues

### Missing BATS Helpers

```
load: cannot determine path to "bats-support/load"
```

**Fix:** Install BATS with helpers:
```bash
brew install bats-core
```

### Missing System Libraries (CI)

```
ImportError: libgomp.so.1: cannot open shared object file
```

**Fix:** Add the library to `Dockerfile.dx-qa`:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    # ... add your library here
```

### Package Data Not Included

```
FileNotFoundError: [Errno 2] No such file or directory: '.../maps/training_facility.map'
```

**Fix:** Check `pyproject.toml` `[tool.setuptools.package-data]` section includes the required files.

## Known Issues

### `cogames docs readme` Fails When Installed from Wheel

The `cogames docs readme` command fails when cogames is installed from a wheel because it uses `Path(__file__).parent.parent.parent` to locate README.md, which points to site-packages instead of the source tree.

**Status:** Known limitation. The docs index (`cogames docs`) works, but fetching specific documents that live outside the package fails.

**Workaround:** Access documentation via the GitHub repo or PyPI page.

**Future fix:** Include docs in `package_data` or use `importlib.resources` for bundled documentation.

## Adding New Tests

When adding new CLI commands or features, add corresponding tests to `dx-qa-test.bats`.

### Test Structure

```bash
@test "descriptive test name" {
    # Run the command
    run cogames your-command --flags

    # Assert success (exit code 0)
    assert_success

    # Assert output contains expected content
    assert_output --partial "expected text"
}
```

### Test Categories

Tests are organized into stages matching the user journey:

1. **Stage 1: Install Verification** - Command exists, help works
2. **Stage 2: Explore** - Discovery commands (missions, variants, etc.)
3. **Stage 3: Validate** - Configuration and environment checks
4. **Stage 4: Play** - Core gameplay functionality
5. **Stage 5: Create** - Policy and mission creation
6. **Stage 6: Run** - Evaluation functionality
7. **Error Handling** - Graceful error messages

### BATS Assert Reference

```bash
# Exit code assertions
assert_success          # Exit code 0
assert_failure          # Non-zero exit code

# Output assertions
assert_output "exact"   # Exact match
assert_output --partial "substring"  # Contains
assert_output --regexp "pattern"     # Regex match

# Line assertions
assert_line "exact line"
assert_line --partial "substring"
assert_line --index 0 "first line"
```

## Directory Structure

```
packages/cogames/devops/dx-qa/
├── README.md              # This file
├── dx-qa-test.bats        # BATS test suite (shared between local and CI)
├── run-dx-qa-local.sh     # Local dev wrapper script
├── Dockerfile.dx-qa       # CI container specification
└── entrypoint.sh          # CI container entrypoint
```

## CI Integration

The DX QA tests integrate into the GitHub Actions workflow. See the release workflow for how the Docker container is built and run as a release gate.

### Integration Point

```yaml
# In .github/workflows/release.yml (example)
- name: Run DX QA tests
  run: |
    docker build -f packages/cogames/devops/dx-qa/Dockerfile.dx-qa \
      -t cogames-dx-qa packages/cogames/devops/dx-qa/
    docker run -v ${{ github.workspace }}/packages/cogames/dist:/dist cogames-dx-qa
```

## Philosophy

**Why clean room testing?**

- Unit tests run in development environments with all dev dependencies
- CI tests run in environments with test fixtures and mocks
- Users run in fresh environments with only the published wheel

DX QA catches issues that slip through other testing:
- Missing package data files
- Incorrect entry point configuration
- Runtime dependencies not declared
- Import errors that only appear without dev dependencies
- CLI commands that fail when run standalone
