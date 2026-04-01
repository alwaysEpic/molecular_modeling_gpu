# Benchmark Results

Best recorded time per test configuration, per processor.

## Default First-Hit (spherical receiver, with drift)

Parameters: `-c -f`, Db=1E-11, deltaT=1E-7, t=1E-3

| Processor | 1,000 paths | 5,000 paths | 10,000 paths | 50,000 paths | 100,000 paths |
|-----------|------------|------------|-------------|-------------|---------------|
| Intel i7-7700K 4.2GHz (thesis) | 1.64s | 8.26s | 16.54s | 82.66s | 166.04s |
| Apple Silicon (M-series) | 1.41s | — | — | — | — |
| Colab Xeon (single core) | 2.67s | — | 28.58s | — | — |
| GTX 1070 (thesis) | 2.27s | 3.46s | 4.81s | 41.52s | 111.98s |
| Tesla T4 (Colab) | 1.43s | — | 10.74s | — | — |

## 1D First-Hit with Drift

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7, 100k steps

| Processor | 10,000 paths |
|-----------|-------------|
| Apple Silicon (M-series) | 134.25s |
| Colab Xeon (single core) | 281.27s |
| Tesla T4 (Colab) | 18.45s |

## Validation

| Test | Backend | Hits | NRMSE | Status |
|------|---------|------|-------|--------|
| 1D first-hit w/ drift, 10k paths | CPU (Apple Silicon) | 9,788 | 0.033 | PASS |
| 1D first-hit w/ drift, 10k paths | GPU (Tesla T4) | 9,755 | 0.040 | PASS |
| 1D first-hit w/ drift, 10k paths | CPU (Colab Xeon) | 9,783 | 0.033 | PASS |

## Notes

- GPU speedup improves significantly with longer simulations (more steps per path).
  Default test (10k steps): 2.66x. 1D limit test (100k steps): 15.2x.
- Known bottleneck: per-timestep cudaMemcpy from GPU to host for hit detection.
- gridSize bug (integer division truncation) affects runs where iter is not a multiple of blockSize.
