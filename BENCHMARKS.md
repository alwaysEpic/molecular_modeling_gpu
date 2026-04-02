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

### GPU vs CPU Speedup (current)

| Paths | CPU (Colab Xeon) | GPU (T4, long) | Speedup |
|-------|-----------------|---------------------|---------|
| 1,000 | 1.79s | 0.018s | **97x** |
| 10,000 | 15.4s | 0.013s | **1,148x** |

## 1D First-Hit with Drift

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7, 100k steps

| Processor | Change | 10,000 paths |
|-----------|--------|-------------|
| Colab Xeon CPU | current | 126s |
| Tesla T4 GPU | original thesis code | 16.7s |
| Tesla T4 GPU | device-side hit detection | 0.633s |
| Tesla T4 GPU | **long kernel** | **0.065s** |

GPU vs CPU speedup: 126s / 0.065s = **1,938x**

## Regression Test (vs thesis-baseline, same Colab session)

| Test | Baseline | Current | Improvement |
|------|----------|---------|-------------|
| 1k default first-hit | 1.334s | 0.008s | **99.4%** |
| 10k default first-hit | 11.214s | 0.008s | **99.9%** |
| 10k 1D limit (100k steps) | 16.684s | 0.039s | **99.8%** |

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
| **Total vs thesis baseline** | **~1,400x** |

## Known Limitations

- **--use_fast_math disabled**: caused wall reflection test failures due to reduced
  trig precision. Could be re-enabled selectively for non-reflection code paths.
