# Learnings

Notes and lessons learned during development.

## Process

- **Always run validation before committing code.** Use `scripts/validate_before_commit.sh` to verify output matches analytical solutions. Even "structural only" changes can introduce bugs.

## Bugs Found

### abs() on float calls integer abs (2026-04-01)
`abs()` from `<stdlib.h>` truncates floats to int silently. At nanometer scales (radius = 8E-6), `abs(0.000003)` becomes `0`. Wall collision detection was completely broken. Use `fabsf()` for float or `fabs()` for double.

### gridSize integer division (2026-04-01)
`ceil(iter/blockSize)` where both are int — the division truncates *before* `ceil` sees it. `ceil(7)` is still `7`. Particles in the last partial block were never simulated. Fix: `(iter + blockSize - 1) / blockSize`.

### Data race on shared device pointers (2026-04-01)
`x_new` and `y_new` in wall reflection were single device-allocated doubles shared across all threads. Multiple threads writing simultaneously produces garbage. Needs per-thread local storage.

### Float/double mismatch in d_reflection (2026-04-01)
Using `powf`, `sqrtf`, `sinf`, `cosf` (float precision) but storing in `double` variables. Gives false confidence in precision — intermediate results are truncated to float. Use matching precision throughout.

## Performance

- GPU speedup scales with simulation length: 2.66x at 10k steps, 15.2x at 100k steps per path.
- Main bottleneck is per-timestep `cudaMemcpy` from GPU to host for hit detection.
- Apple Silicon M-series is ~2x faster than Colab Xeon for single-threaded CPU work.
