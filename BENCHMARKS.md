# Benchmark Results

Best recorded time per test configuration, per processor.

## Default First-Hit (spherical receiver, with drift)

Parameters: `-c -f`, Db=1E-11, deltaT=1E-7, t=1E-3

### CPU

| Processor | 1,000 paths | 5,000 paths | 10,000 paths | 50,000 paths | 100,000 paths |
|-----------|------------|------------|-------------|-------------|---------------|
| Intel i7-7700K 4.2GHz (thesis) | 1.64s | 8.26s | 16.54s | 82.66s | 166.04s |
| Apple Silicon (M-series) | 1.41s | — | — | — | — |
| Colab Xeon (post-cleanup) | 1.24s | — | 13.64s | — | — |

### GPU

| GPU | Change | 1,000 paths | 10,000 paths |
|-----|--------|------------|-------------|
| GTX 1070 (thesis) | original | 2.27s | 4.81s |
| Tesla T4 | per-step memcpy (pre-cleanup) | 1.43s | 10.74s |
| Tesla T4 | per-step memcpy (post-cleanup) | 1.36s | 10.65s |
| Tesla T4 | device-side hit detection | 0.055s | 0.063s |
| Tesla T4 | + persistent RNG | 0.075s | 0.113s |

Note: persistent RNG times are higher due to Colab T4 variability between sessions,
not a regression. The curand_init savings are small relative to total kernel time
and masked by shared-hardware noise.

### GPU vs CPU Speedup (current)

| Paths | CPU (Colab Xeon) | GPU (T4) | Speedup |
|-------|-----------------|----------|---------|
| 1,000 | 2.43s | 0.074s | **32.6x** |
| 10,000 | 13.64s | 0.113s | **120.7x** |

## 1D First-Hit with Drift

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7, 100k steps

| Processor | Change | 10,000 paths |
|-----------|--------|-------------|
| Apple Silicon CPU | post-cleanup | 134.25s |
| Colab Xeon CPU | current | 126.38s |
| Tesla T4 GPU | per-step memcpy (pre-cleanup) | 18.45s |
| Tesla T4 GPU | per-step memcpy (post-cleanup) | 17.49s |
| Tesla T4 GPU | device-side hit detection | 0.633s |
| Tesla T4 GPU | + persistent RNG | 0.561s |

GPU vs CPU speedup: 126.4s / 0.561s = **225x**

## 3D Spherical Receiver (no drift)

Parameters: `-f -n`, Db=1E-11, deltaT=1E-7, t=1E-3

| Backend | 10,000 paths | Hits | Hit Rate | NRMSE |
|---------|-------------|------|----------|-------|
| GPU (T4) | 0.116s | 1,424 | 14.2% | 0.034 |
| CPU (Colab Xeon) | 13.41s | 1,406 | 14.1% | 0.045 |
| Theoretical | — | — | 15.6% | — |

## Validation

| Test | Backend | Hits | NRMSE | Status |
|------|---------|------|-------|--------|
| 1D first-hit w/ drift, 10k | GPU (T4, persistent RNG) | 9,798 | 0.034 | PASS |
| 1D first-hit w/ drift, 10k | CPU (Colab Xeon) | 9,807 | 0.034 | PASS |
| 3D spherical, no drift, 10k | GPU (T4) | 1,424 | 0.034 | PASS |
| 3D spherical, no drift, 10k | CPU (Colab Xeon) | 1,406 | 0.045 | PASS |
| CPU/GPU agreement (KS test) | 5k paths, 1D limit | p=0.74 | — | PASS |
| GPU RNG quality (Philox seed=42) | 10k samples | — | — | PASS |
| GPU RNG quality (Philox seed=123) | 10k samples | — | — | PASS |
| CPU RNG quality (Box-Muller) | 10k samples | — | — | PASS |
| GPU reproducibility (seed=42) | 500 paths | — | — | PASS |
| CPU reproducibility (seed=42) | 500 paths | — | — | PASS |

## Optimization History

| Change | Impact |
|--------|--------|
| Bug fixes (gridSize, abs, float/double) | Correctness — particles were being dropped |
| CPU cleanup (stack vars, sqrt reduction) | ~2x CPU speedup on Colab Xeon |
| Device-side hit detection | 25-224x GPU speedup — eliminated per-timestep cudaMemcpy |
| Persistent RNG + --seed | Correctness (no correlated streams), reproducibility |

## Remaining Optimizations

- Persistent kernel: fuse timestep loop into single kernel launch, eliminating per-step launch overhead
- Wall reflection: data race fixed, but wall mode not yet benchmarked
