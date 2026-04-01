# Benchmark Results

Best recorded time per test configuration, per processor.

## Default First-Hit (spherical receiver, with drift)

Parameters: `-c -f`, Db=1E-11, deltaT=1E-7, t=1E-3

### CPU

| Processor | 1,000 paths | 5,000 paths | 10,000 paths | 50,000 paths | 100,000 paths |
|-----------|------------|------------|-------------|-------------|---------------|
| Intel i7-7700K 4.2GHz (thesis) | 1.64s | 8.26s | 16.54s | 82.66s | 166.04s |
| Apple Silicon (M-series) | 1.41s | — | — | — | — |
| Colab Xeon (post-cleanup) | 1.24s | — | 14.06s | — | — |

### GPU

| GPU | Change | 1,000 paths | 10,000 paths |
|-----|--------|------------|-------------|
| GTX 1070 (thesis) | original | 2.27s | 4.81s |
| Tesla T4 | pre-cleanup (per-step memcpy) | 1.43s | 10.74s |
| Tesla T4 | post-cleanup (per-step memcpy) | 1.36s | 10.65s |
| Tesla T4 | **device-side hit detection** | **0.055s** | **0.063s** |

### GPU vs CPU Speedup

| Paths | CPU (Colab Xeon) | GPU (T4, device-side hit) | Speedup |
|-------|-----------------|--------------------------|---------|
| 1,000 | 1.49s | 0.055s | **26.9x** |
| 10,000 | 14.06s | 0.063s | **223.8x** |

## 1D First-Hit with Drift

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7, 100k steps

| Processor | Change | 10,000 paths |
|-----------|--------|-------------|
| Apple Silicon CPU | post-cleanup | 134.25s |
| Colab Xeon CPU | post-cleanup | 125.60s |
| Tesla T4 GPU | pre-cleanup (per-step memcpy) | 18.45s |
| Tesla T4 GPU | post-cleanup (per-step memcpy) | 17.49s |
| Tesla T4 GPU | **device-side hit detection** | **0.633s** |

GPU vs CPU speedup: 125.6s / 0.633s = **198x**

## Validation

| Test | Backend | Hits | NRMSE | Status |
|------|---------|------|-------|--------|
| 1D first-hit w/ drift, 10k paths | CPU (Apple Silicon) | 9,788 | 0.033 | PASS |
| 1D first-hit w/ drift, 10k paths | CPU (Colab Xeon) | 9,778 | 0.039 | PASS |
| 1D first-hit w/ drift, 10k paths | GPU (T4, per-step memcpy) | 9,801 | 0.035 | PASS |
| 1D first-hit w/ drift, 10k paths | GPU (T4, device-side hit) | 9,798 | 0.042 | PASS |

## Optimization History

| Change | Impact |
|--------|--------|
| Bug fixes (gridSize, abs, float/double) | Correctness — particles were being dropped |
| CPU cleanup (stack vars, sqrt reduction) | ~2x CPU speedup on Colab Xeon |
| **Device-side hit detection** | **25-224x GPU speedup** — eliminated per-timestep cudaMemcpy |

## Remaining Optimizations

- RNG: `curand_init` called every kernel invocation — persistent state would save ~100 cycles/thread/step
- Persistent kernel: fuse timestep loop into single kernel launch, eliminating per-step launch overhead
- Wall reflection data race fixed (thread-local stack vars), but wall mode not yet benchmarked
