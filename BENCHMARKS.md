# Benchmark Results

Best recorded time per test configuration, per processor.

## 1D First-Hit (with drift)

Parameters: `-f -l 3E-7 -t 1E-2`, Db=1E-11, vel=1E-4, deltaT=1E-7

| Processor | 1,000 paths | 10,000 paths | 100,000 paths |
|-----------|------------|-------------|---------------|
| Intel i7-7700K 4.2GHz (thesis, single core) | 1.67s | 16.54s | 166.04s |
| Apple Silicon (M-series, single core) | — | 134.25s | — |

## Default First-Hit (spherical receiver, no drift)

Parameters: `-f -n`, Db=1E-11, deltaT=1E-7, t=1E-3, rec_rad=10nm, rec_dist=50nm

| Processor | 1,000 paths | 5,000 paths | 10,000 paths |
|-----------|------------|------------|-------------|
| Intel i7-7700K 4.2GHz (thesis, single core) | 1.67s | 8.24s | 16.41s |
| Apple Silicon (M-series, single core) | 1.41s | — | — |

## GPU Comparison (first-hit, with drift)

Parameters: `-c -f`, Db=1E-11, deltaT=1E-7, t=1E-3

| GPU | 10,000 paths | 50,000 paths | 100,000 paths |
|-----|-------------|-------------|---------------|
| GTX 1070 (thesis) | 4.81s | 41.52s | 111.98s |

## Validation

| Test | NRMSE | Status |
|------|-------|--------|
| 1D first-hit w/ drift (CPU, 10k paths, Apple Silicon) | 0.033 | PASS |
