# Benchmark Results

Best recorded time per test configuration, per processor.

## Default First-Hit (spherical receiver, with drift)

Parameters: `-c -f`, Db=1E-11, deltaT=1E-7, t=1E-3

| Processor | 1,000 paths | 5,000 paths | 10,000 paths | 50,000 paths | 100,000 paths |
|-----------|------------|------------|-------------|-------------|---------------|
| Intel i7-7700K 4.2GHz (thesis) | 1.64s | 8.26s | 16.54s | 82.66s | 166.04s |
| Apple Silicon (M-series) | 1.41s | — | — | — | — |
| Colab Xeon (pre-cleanup) | 2.67s | — | 28.58s | — | — |
| Colab Xeon (post-cleanup) | 1.24s | — | 13.29s | — | — |
| GTX 1070 (thesis) | 2.27s | 3.46s | 4.81s | 41.52s | 111.98s |
| Tesla T4 (pre-cleanup) | 1.43s | — | 10.74s | — | — |
| Tesla T4 (post-cleanup) | 1.36s | — | 10.65s | — | — |

## 1D First-Hit with Drift

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7, 100k steps

| Processor | 10,000 paths |
|-----------|-------------|
| Apple Silicon (M-series) | 134.25s |
| Colab Xeon (pre-cleanup) | 281.27s |
| Colab Xeon (post-cleanup) | 124.70s |
| Tesla T4 (pre-cleanup) | 18.45s |
| Tesla T4 (post-cleanup) | 17.49s |

## Validation

| Test | Backend | Hits | NRMSE | Status |
|------|---------|------|-------|--------|
| 1D first-hit w/ drift, 10k paths | CPU (Apple Silicon) | 9,788 | 0.033 | PASS |
| 1D first-hit w/ drift, 10k paths | GPU (Tesla T4, pre-cleanup) | 9,755 | 0.040 | PASS |
| 1D first-hit w/ drift, 10k paths | CPU (Colab Xeon, pre-cleanup) | 9,783 | 0.033 | PASS |
| 1D first-hit w/ drift, 10k paths | GPU (Tesla T4, post-cleanup) | 9,801 | 0.035 | PASS |
| 1D first-hit w/ drift, 10k paths | CPU (Colab Xeon, post-cleanup) | 9,778 | 0.039 | PASS |

## Notes

- CPU cleanup (stack vars, reduced redundant sqrt calls) gave ~2x speedup on Colab Xeon.
- GPU speedup improves with longer simulations (more steps per path).
  Default test (10k steps): 1.25x. 1D limit test (100k steps): 7.1x.
- gridSize bug fixed — was dropping particles past last full block.
- Known bottleneck: per-timestep cudaMemcpy from GPU to host for hit detection.
