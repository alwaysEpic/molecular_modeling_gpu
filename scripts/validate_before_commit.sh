#!/bin/bash
# Run validation tests before committing.
# Usage: ./scripts/validate_before_commit.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_DIR/build"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Python venv not found. Run:"
    echo "  python3 -m venv scripts/.venv && scripts/.venv/bin/pip install -r scripts/requirements.txt"
    exit 1
fi

# Build
echo "=== Building ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) 2>&1 | tail -5

# Run CPU validation (1D first-hit with drift, 1000 paths — fast)
echo ""
echo "=== CPU Validation (1k paths, ~1.5s) ==="
./mc_sim_cpu -i 1000 -f -l 3E-7 -t 1E-2

echo ""
"$VENV_PYTHON" "$SCRIPT_DIR/validate_1d_firsthit.py" output_h.csv \
    --no-plot --dist 3E-7 --vel 1E-4 --timestep 1E-7 --timestop 1E-2

# Run GPU validation if mc_sim exists
if [ -f ./mc_sim ]; then
    echo ""
    echo "=== GPU Validation (1k paths) ==="
    ./mc_sim -i 1000 -f -l 3E-7 -t 1E-2

    echo ""
    "$VENV_PYTHON" "$SCRIPT_DIR/validate_1d_firsthit.py" output_d_wide.csv \
        --no-plot --dist 3E-7 --vel 1E-4 --timestep 1E-7 --timestop 1E-2
fi

echo ""
echo "=== All validations complete ==="
