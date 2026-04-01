# Learnings

Notes and lessons learned during development.

## Process

- **Always run validation before committing code.** Use `scripts/validate_before_commit.sh` (8 CPU tests) and the Colab notebook (14 tests including GPU) to verify output matches analytical solutions. Even "structural only" changes can introduce bugs.
- **Colab T4 timing varies between sessions** — shared hardware means 20-50% variation is normal. Compare trends, not individual numbers.
- **Stale binaries on Colab** — if you `%cd` into a directory from a previous clone without rebuilding, you'll run old code. Always rebuild after a fresh clone.

## Bugs Found

### abs() on float calls integer abs (2026-04-01)
`abs()` from `<stdlib.h>` truncates floats to int silently. At nanometer scales (radius = 8E-6), `abs(0.000003)` becomes `0`. Wall collision detection was completely broken. Use `fabsf()` for float or `fabs()` for double.

### gridSize integer division (2026-04-01)
`ceil(iter/blockSize)` where both are int — the division truncates *before* `ceil` sees it. `ceil(7)` is still `7`. Particles in the last partial block were never simulated. Fix: `(iter + blockSize - 1) / blockSize`.

### Data race on shared device pointers (2026-04-01)
`x_new` and `y_new` in wall reflection were single device-allocated doubles shared across all threads. Multiple threads writing simultaneously produces garbage. Fixed with per-thread stack-local variables.

### Float/double mismatch in d_reflection (2026-04-01)
Using `powf`, `sqrtf`, `sinf`, `cosf` (float precision) but storing in `double` variables. Gives false confidence in precision — intermediate results are truncated to float. Use matching precision throughout.

### RNG: per-call clock64() is faster than persistent state (2026-04-01)
The thesis approach of `curand_init(clock64(), idx, 0, &state)` per kernel call was deliberately chosen and is **4-5x faster** than persistent RNG state on a T4. A/B test results (10k particles × 10k steps, same session):
- Per-call clock64(): ~27ms
- Persistent state (global memory): ~117ms

**Why:** Philox state created on the stack lives in registers (~1 cycle access). Persistent state in global memory costs ~400-800 cycles per read/write. The `curand_init` cost for Philox is cheap (counter-based, just sets a few integers) — much less than the global memory round-trip of 64 bytes × 2 (read+write) × 10k threads per timestep.

The thesis breakthrough was two things working together:
1. Choosing Philox (cheap init) over XORWOW (expensive init)
2. Using `clock64()` for fresh seeds without persistent state

Trade-off: `clock64()` can produce correlated streams when threads launch at the same clock tick, but thesis validation (cross-correlation vs MATLAB randn) confirmed acceptable quality for the physics. When `--seed` is provided, we use the fixed seed instead of `clock64()` for reproducibility — same performance, deterministic output.

## Performance

- **Device-side hit detection was the biggest win** — moving hit checks into the kernel and eliminating per-timestep cudaMemcpy gave 25-224x GPU speedup.
- GPU speedup scales with simulation length: more timesteps = more amortized kernel launch overhead.
- CPU cleanup (stack vars, reduced redundant sqrt calls) gave ~2x CPU speedup.
- Apple Silicon M-series is ~2x faster than Colab Xeon for single-threaded CPU work.
- **Per-call Philox RNG with clock64() is 4-5x faster than persistent state** — register-local stack state beats global memory round-trips. Don't assume "initialize once" is always faster; memory hierarchy matters more than init cost for lightweight RNGs.
