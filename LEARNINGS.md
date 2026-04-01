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

## Validation Methodology (2026-04-01)

### KS test over NRMSE for simulation validation
NRMSE on histograms is bin-dependent (change binning, change result), has no known null distribution, and is not a formal statistical test. The Kolmogorov-Smirnov test compares empirical CDF against theoretical CDF with no binning, provides a rigorous p-value, and is consistent. NRMSE kept as supplementary human-readable metric. References: NIST Handbook, Law (2015), Devroye (1986).

### 3D hit-time validation requires two separate tests
The 3D H_Diff PDF integrates to Pr(hit) = a/d < 1, not to 1. Observed hitting times are samples from the *conditional* distribution (given that the particle hit). KS test must compare against H_Diff normalized to integrate to 1, not the raw PDF. Hit probability validated separately with a binomial test. This is standard practice: Noel et al. (2014), Yilmaz et al. (2014).

### alpha = 0.001 for CI, not 0.05
At alpha = 0.05, ~5% of CI runs false-fail even with correct code. At alpha = 0.001, false positive rate is 0.1%. With n=10,000 the KS test still detects CDF shifts of ~0.019, catching any real bug.

### 1D inverse Gaussian has exact scipy parameterization
No numerical integration needed. `scipy.stats.invgauss` with `mu = 2*D/(b*v)`, `scale = b^2/(2*D)` gives exact CDF. Derivation: our PDF `p(t) = b/sqrt(4*pi*D*t^3) * exp(-(v*t-b)^2/(4*D*t))` is a standard inverse Gaussian with mean `mu = b/v` and shape `lambda = b^2/(2*D)`. scipy's parameterization absorbs lambda into scale.

### 3D conditional CDF via numerical integration
The 3D H_Diff PDF integrates to Pr(hit) < 1, but our samples are only from particles that hit. We must compare against H_Diff normalized to 1 (conditional distribution). CDF built by `cumulative_trapezoid` on a 500k-point grid, then normalized. `interp1d` creates a callable CDF for `scipy.stats.kstest`. This is standard: see Noel et al. (2014), Jamali et al. (2019).

### Thesis equation 4.3 has a typesetting error
The PDF shows `-(vt - b^2)` but should be `-(v*t - b)^2`. All code (MATLAB and Python) implements the correct form. The squared term applies to the full expression `(v*t - b)`, not just `b`. Verified against Kadloor et al. 2012 Eq. 12.

### Per-call Philox RNG with clock64() is 4-5x faster than persistent state
See detailed entry under Bugs Found.

## Remaining Validation Improvements

Items identified during deep review, to be addressed:
1. ~~KS test as formal pass/fail~~ (done — 1D uses scipy.stats.invgauss, 3D uses numerical CDF + binomial hit rate test)
2. CSV NaN filtering — scripts don't skip headers, MATLAB skips 2 rows
3. ~~np.trapezoid compatibility~~ (done — pinned numpy>=2.0, scipy>=1.11 in requirements.txt)
4. ~~Dead QQ plot code in validate_rng.py~~ (done — removed dead code, dropped try/except for scipy since it's pinned)
5. 3D test at 1k paths — only ~200 hits across 120 bins is meaningless
6. Wall reflection validation — TobyThesisTest_walls.m not ported
7. Multi-parameter 1D test — TobyThesisTest_dist.m uses different params
8. Timestep convergence test — no test validates dt sensitivity
9. Inter-stream RNG correlation — test cross-thread correlation
10. Edge cases: empty CSV, single hit, degenerate histograms

## Performance

- **Device-side hit detection was the biggest win** — moving hit checks into the kernel and eliminating per-timestep cudaMemcpy gave 25-224x GPU speedup.
- GPU speedup scales with simulation length: more timesteps = more amortized kernel launch overhead.
- CPU cleanup (stack vars, reduced redundant sqrt calls) gave ~2x CPU speedup.
- Apple Silicon M-series is ~2x faster than Colab Xeon for single-threaded CPU work.
- **Per-call Philox RNG with clock64() is 4-5x faster than persistent state** — register-local stack state beats global memory round-trips. Don't assume "initialize once" is always faster; memory hierarchy matters more than init cost for lightweight RNGs.
