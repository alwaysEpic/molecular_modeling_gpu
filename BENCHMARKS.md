# Benchmark Results

Best recorded time per test configuration, per processor.

## Default First-Hit (spherical receiver, with drift)

Parameters: `-f`, Db=1E-11, deltaT=1E-7, t=1E-3

### CPU

| Processor | 1,000 paths | 10,000 paths |
|-----------|------------|-------------|
| Intel i7-7700K 4.2GHz (thesis) | 1.64s | 16.54s |
| Apple Silicon (M-series) | 1.41s | — |
| Colab Xeon | 1.25s | 13.7s |

### GPU

| GPU | Change | 1,000 paths | 10,000 paths |
|-----|--------|------------|-------------|
| GTX 1070 (thesis) | original | 2.27s | 4.81s |
| Tesla T4 | original thesis code | 1.38s | 11.2s |
| Tesla T4 | device-side hit detection | 0.055s | 0.063s |
| Tesla T4 | + precompute/curand_normal4/block256 | 0.012s | 0.013s |
| Tesla T4 | **+ long kernel** | **0.008s** | **0.008s** |

### GPU vs CPU Speedup (A100)

| Paths | CPU (Colab Xeon) | GPU (A100, long) | Speedup |
|-------|-----------------|---------------------|---------|
| 1,000 | 1.39s | 0.056s | **25x** |
| 10,000 | 138.9s | 0.068s | **2,037x** |

## 1D First-Hit with Drift

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7, 100k steps

| Processor | Change | 10,000 paths |
|-----------|--------|-------------|
| Colab Xeon CPU | current | 139s |
| Tesla T4 GPU | original thesis code | 16.7s |
| Tesla T4 GPU | device-side hit detection | 0.633s |
| Tesla T4 GPU | **long kernel** | **0.065s** |
| **A100 GPU** | **long kernel** | **0.068s** |

GPU vs CPU speedup (A100): 139s / 0.068s = **2,037x**

## Regression Test (vs thesis-baseline, A100)

| Test | Baseline | Current | Improvement |
|------|----------|---------|-------------|
| 1k default first-hit | 5.87s | 0.067s | **98.9%** |
| 10k default first-hit | 16.61s | 0.068s | **99.6%** |
| 10k 1D limit (100k steps) | 16.61s | 0.068s | **99.6%** |

## Scale Sweep (A100, long kernel, default first-hit, 10k steps)

| Paths | Time | Thesis 1070 |
|-------|------|-------------|
| 10,000 | 0.012s | — |
| 50,000 | 0.016s | — |
| 100,000 | 0.026s | — |
| 500,000 | 0.101s | — |
| 1,000,000 | 0.192s | — |
| 2,500,000 | 0.462s | — |
| 5,000,000 | 0.914s | 5.08s |

Throughput: **5.5M paths/sec** (A100) vs thesis 1M paths/sec (GTX 1070).

## Gold Standard (A100, long kernel, dt=1E-8)

| Test | Particles | Steps | Time | KS p-value |
|------|-----------|-------|------|------------|
| 1D first-hit | 10,000,000 | 1,000,000 | **143s** | 0.047 (PASS) |

10 trillion particle-steps in under 3 minutes.

## Validation

| Test | Backend | Result | Notes |
|------|---------|--------|-------|
| 1D first-hit KS (1k paths) | GPU | PASS | |
| 1D first-hit KS (1k paths) | CPU | PASS | |
| 1D first-hit KS (10k paths) | GPU | PASS | After Brownian bridge correction |
| 3D spherical KS (10k paths) | GPU | PASS | |
| 3D spherical binomial (10k) | GPU | PASS | After Brownian bridge correction |
| 3D spherical (1k paths) | CPU | PASS | |
| CPU/GPU agreement (5k paths) | Both | PASS | KS p > 0.01 |
| GPU RNG quality (Philox) | GPU | PASS | |
| CPU RNG quality (Box-Muller) | CPU | PASS | |
| GPU reproducibility (seed=42) | GPU | PASS | Identical output |
| CPU reproducibility (seed=42) | CPU | PASS | Identical output |
| Wall reflection (10k paths) | GPU | PASS | |
| Performance regression | GPU | PASS | No regression vs baseline |

## Optimization History

| Change | GPU Impact |
|--------|-----------|
| Bug fixes (gridSize, abs, float/double) | Correctness |
| CPU cleanup (stack vars, sqrt reduction) | ~2x CPU speedup |
| Device-side hit detection | 25-224x — eliminated per-timestep cudaMemcpy |
| Precompute constants + curand_normal4 + block 256 | ~5x on top of above |
| **Long kernel (d_simulate_isolated)** | **~3x on top of above** |
| Brownian bridge correction | Correctness (fixes boundary-crossing bias) |
| Wide kernel (d_update) with same optimizations | Same improvements, global memory positions |
| CUDA Graphs for wide kernel | Reverted — added overhead, no improvement |
| **Total vs thesis wide baseline** | **~2,000x** |

## Known Limitations

- **--use_fast_math disabled**: caused wall reflection test failures due to reduced
  trig precision. Could be re-enabled selectively for non-reflection code paths.
