#!/bin/bash
# Pre-commit validation suite.
# Runs all tests that can execute locally (CPU-only).
# GPU tests require Colab — see colab_build_test.ipynb.
#
# Usage: ./scripts/validate_before_commit.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_DIR/build"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

PASS=0
FAIL=0
SKIP=0

report() {
    local status=$1
    local name=$2
    if [ "$status" = "PASS" ]; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    elif [ "$status" = "FAIL" ]; then
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    else
        echo "  [SKIP] $name"
        SKIP=$((SKIP + 1))
    fi
}

# Check venv
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Python venv not found. Run:"
    echo "  python3 -m venv scripts/.venv && scripts/.venv/bin/pip install -r scripts/requirements.txt"
    exit 1
fi

# Build
echo "=== Build ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5
echo ""

# Test 1: CPU 1D first-hit validation (KS test)
echo "=== Test 1: CPU 1D First-Hit with Drift (1k paths) ==="
./mc_sim_cpu -i 1000 -f -l 3E-7 -t 1E-2 > /dev/null 2>&1
RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_1d_firsthit.py" output_h.csv \
    --no-plot --dist 3E-7 --vel 1E-4 --timestep 1E-7 --timestop 1E-2 2>&1 | grep -E "^PASS|^FAIL")
if echo "$RESULT" | grep -q "^PASS"; then
    report "PASS" "1D first-hit validation (CPU, KS test, 1k paths)"
else
    report "FAIL" "1D first-hit validation (CPU, KS test, 1k paths)"
fi

# Test 1b: CPU 1D first-hit with different params (matching TobyThesisTest_dist.m)
# dist=1E-7, vel=3E-4, timestep=1E-8 — tests different Peclet number regime
echo "=== Test 1b: CPU 1D First-Hit Alt Params (1k paths, dt=1E-8) ==="
./mc_sim_cpu -i 1000 -f -l 1E-7 -t 1E-3 -d 1E-8 --velocity 3E-4 > /dev/null 2>&1
RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_1d_firsthit.py" output_h.csv \
    --no-plot --dist 1E-7 --vel 3E-4 --timestep 1E-8 --timestop 1E-3 2>&1 | grep -E "^PASS|^FAIL")
if echo "$RESULT" | grep -q "^PASS"; then
    report "PASS" "1D first-hit alt params (CPU, KS test, 1k paths, dt=1E-8)"
else
    report "FAIL" "1D first-hit alt params (CPU, KS test, 1k paths, dt=1E-8)"
fi

# Test 2: CPU 3D spherical receiver validation (KS + binomial)
echo "=== Test 2: CPU 3D Spherical Receiver (1k paths, no drift) ==="
./mc_sim_cpu -i 1000 -f -n > /dev/null 2>&1
RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_3d_diffusion.py" output_h.csv \
    --no-plot --total-paths 1000 2>&1 | grep -E "^PASS|^FAIL")
if echo "$RESULT" | grep -q "^PASS"; then
    report "PASS" "3D diffusion-only validation (CPU, KS + binomial, 1k paths)"
else
    report "FAIL" "3D diffusion-only validation (CPU, KS + binomial, 1k paths)"
fi

# Test 2b: Position distribution validation (D-estimation)
echo "=== Test 2b: CPU Position Distribution (2k particles) ==="
RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_position_distribution.py" --binary "$BUILD_DIR/mc_sim_cpu" --no-plot 2>&1 | grep -E "^PASS|^FAIL")
if echo "$RESULT" | grep -q "^PASS"; then
    report "PASS" "Position distribution (CPU, D-estimation, 2k particles)"
else
    report "FAIL" "Position distribution (CPU, D-estimation, 2k particles)"
fi

# Test 3: CPU RNG quality
echo "=== Test 3: CPU RNG Quality ==="
if [ -f "$BUILD_DIR/dump_rng_cpu" ] || g++ -O2 -o "$BUILD_DIR/dump_rng_cpu" "$SCRIPT_DIR/dump_rng_cpu.cpp" -lm 2>/dev/null; then
    "$BUILD_DIR/dump_rng_cpu" 10000 > "$BUILD_DIR/rng_test.csv"
    RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_rng.py" "$BUILD_DIR/rng_test.csv" --no-plot 2>&1 | grep -E "^(PASS|FAIL) — Overall")
    if echo "$RESULT" | grep -q "^PASS"; then
        report "PASS" "CPU RNG quality (10k samples)"
    else
        report "FAIL" "CPU RNG quality (10k samples)"
    fi
else
    report "SKIP" "CPU RNG quality (could not compile dump_rng_cpu)"
fi

# Test 4: Edge cases
echo "=== Test 4: Edge Cases ==="

# Single iteration
./mc_sim_cpu -i 1 -f -l 3E-7 -t 1E-3 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    report "PASS" "Single iteration (iter=1)"
else
    report "FAIL" "Single iteration (iter=1)"
fi

# Iter not multiple of blocksize (129 particles, blocksize 128)
./mc_sim_cpu -i 129 -f -l 3E-7 -t 1E-3 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    report "PASS" "Non-aligned iteration count (iter=129)"
else
    report "FAIL" "Non-aligned iteration count (iter=129)"
fi

# No drift
./mc_sim_cpu -i 100 -f -n > /dev/null 2>&1
if [ $? -eq 0 ]; then
    report "PASS" "No drift mode"
else
    report "FAIL" "No drift mode"
fi

# Zero inputs should fail gracefully
set +e
./mc_sim_cpu -i 0 > /dev/null 2>&1
RC=$?
set -e
if [ $RC -ne 0 ]; then
    report "PASS" "Reject iter=0"
else
    report "FAIL" "Reject iter=0 (should exit with error)"
fi

set +e
./mc_sim_cpu -t 0 > /dev/null 2>&1
RC=$?
set -e
if [ $RC -ne 0 ]; then
    report "PASS" "Reject time=0"
else
    report "FAIL" "Reject time=0 (should exit with error)"
fi

# GPU tests (only if mc_sim exists)
echo ""
if [ -f "$BUILD_DIR/mc_sim" ]; then
    # Deep (persistent) kernel — analytical validation
    echo "=== Test 5: GPU Deep Kernel — 1D KS (1k paths) ==="
    ./mc_sim -i 1000 -f -l 3E-7 -t 1E-2 > /dev/null 2>&1
    RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_1d_firsthit.py" output_gpu.csv \
        --no-plot --dist 3E-7 --vel 1E-4 --timestep 1E-7 --timestop 1E-2 2>&1 | grep -E "^PASS|^FAIL")
    if echo "$RESULT" | grep -q "^PASS"; then
        report "PASS" "GPU long kernel — 1D KS (1k paths)"
    else
        report "FAIL" "GPU long kernel — 1D KS (1k paths)"
    fi

    # CPU vs long kernel agreement
    echo "=== Test 6: CPU vs GPU Deep Agreement ==="
    ./mc_sim -i 1000 -c -f -l 3E-7 -t 1E-2 > /dev/null 2>&1
    RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_cpu_gpu_agreement.py" output_h.csv output_gpu.csv \
        --no-plot --timestep 1E-7 2>&1 | grep -E "^(PASS|FAIL) — CPU/GPU")
    if echo "$RESULT" | grep -q "^PASS"; then
        report "PASS" "CPU vs GPU long agreement"
    else
        report "FAIL" "CPU vs GPU long agreement"
    fi

    # Wide (per-step) kernel — analytical validation
    echo "=== Test 7: GPU Wide Kernel — 1D KS (1k paths) ==="
    ./mc_sim -i 1000 -f -l 3E-7 -t 1E-2 -W > /dev/null 2>&1
    RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_1d_firsthit.py" output_gpu.csv \
        --no-plot --dist 3E-7 --vel 1E-4 --timestep 1E-7 --timestop 1E-2 2>&1 | grep -E "^PASS|^FAIL")
    if echo "$RESULT" | grep -q "^PASS"; then
        report "PASS" "GPU wide kernel — 1D KS (1k paths)"
    else
        report "FAIL" "GPU wide kernel — 1D KS (1k paths)"
    fi

    # CPU vs wide kernel agreement
    echo "=== Test 8: CPU vs GPU Wide Agreement ==="
    ./mc_sim -i 1000 -c -f -l 3E-7 -t 1E-2 -W > /dev/null 2>&1
    RESULT=$("$VENV_PYTHON" "$SCRIPT_DIR/validate_cpu_gpu_agreement.py" output_h.csv output_gpu.csv \
        --no-plot --timestep 1E-7 2>&1 | grep -E "^(PASS|FAIL) — CPU/GPU")
    if echo "$RESULT" | grep -q "^PASS"; then
        report "PASS" "CPU vs GPU wide agreement"
    else
        report "FAIL" "CPU vs GPU wide agreement"
    fi
else
    report "SKIP" "GPU long kernel (CUDA not available)"
    report "SKIP" "CPU vs GPU long agreement (CUDA not available)"
    report "SKIP" "GPU wide kernel (CUDA not available)"
    report "SKIP" "CPU vs GPU wide agreement (CUDA not available)"
fi

# Summary
echo ""
echo "=============================="
echo "  PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
echo "=============================="

if [ $FAIL -gt 0 ]; then
    echo "SOME TESTS FAILED — do not commit without investigating"
    exit 1
else
    echo "All tests passed. Safe to commit."
    exit 0
fi
