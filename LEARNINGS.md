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

### RNG re-initialization per kernel call (2026-04-01)
`curand_init(clock64(), idx, 0, &state)` called every timestep, every thread. This was a deliberate thesis-era solution, not an oversight — switching from `curandState_t` (XORWOW, ~48 byte state, slow init) to `curandStatePhilox4_32_10_t` (small state, fast init) combined with `clock64()` seeding was a breakthrough that significantly improved performance. The `clock64()` seed ensured each kernel call got fresh randomness without needing to manage persistent state across calls. It was validated with cross-correlation analysis against MATLAB's randn. Fixed with persistent curandState array allocated once and passed to kernel — same Philox generator, just initialized once instead of per-call. Validated against thesis RNG tests (moments, autocorrelation, Anderson-Darling normality).

## Performance

- **Device-side hit detection was the biggest win** — moving hit checks into the kernel and eliminating per-timestep cudaMemcpy gave 25-224x GPU speedup.
- GPU speedup scales with simulation length: more timesteps = more amortized kernel launch overhead.
- CPU cleanup (stack vars, reduced redundant sqrt calls) gave ~2x CPU speedup.
- Apple Silicon M-series is ~2x faster than Colab Xeon for single-threaded CPU work.
- Persistent RNG saves ~100 cycles/thread/step but is small relative to total kernel time at current scale. Will matter more with persistent kernel optimization.
