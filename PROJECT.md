# Molecular Communication GPU Simulator

## Overview
GPU-accelerated Monte Carlo simulation of molecular communication in blood vessels.
Simulates particle diffusion via Brownian motion with optional laminar drift, modeling
how nanoscale messenger molecules move through the bloodstream.

Based on a master's thesis applying CUDA parallelism to molecular communication research.

## Project Structure
```
src/
  common/
    params.h              # SimParams struct — all simulation constants
    cli.h                 # CLI parsing, usage, verbose output
  main.cu                 # GPU entry point — dispatches to CPU/GPU simulation
  main_cpu.cpp            # CPU-only entry point — no CUDA dependency
  simulation_cpu.cpp/.h   # CPU reference: Brownian motion, hit detection
  simulation_gpu.cu/.h    # GPU kernels: d_update, d_reflection
  timing.c/.h             # Wall-clock timing utility
scripts/
  validate_1d_firsthit.py # Validates output against analytical solution (thesis eq 4.3)
  validate_before_commit.sh # Pre-commit build + validation
  colab_build_test.ipynb  # Google Colab notebook for GPU testing
  requirements.txt        # Python deps (numpy, matplotlib)
matlab/                   # Reference MATLAB Fokker-Planck implementations
docs/thesis.pdf           # Full thesis document
```

## Physics Model
- Brownian motion: `x += sqrt(2 * Db * deltaT) * randn` for each axis
- Laminar drift: `z += velocity * deltaT` (z-direction only)
- Wall reflection: parametric line-circle intersection in cylindrical blood vessel
- Collision detection: spherical receiver or 1D planar limit

## Key Parameters
- `Db` = 1E-11 m^2/s (diffusion coefficient)
- `velocity` = 1E-4 m/s (laminar flow)
- `deltaT` = 1E-7 s (time step)
- `radius` = 8E-6 m (blood vessel radius)
- GPU RNG: curandStatePhilox4_32_10_t

## Build
```bash
mkdir build && cd build
cmake ..                                    # auto-detects CUDA
make                                        # builds mc_sim_cpu, and mc_sim if CUDA found
```
To target a specific GPU architecture:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75      # Turing (T4, RTX 2080)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86      # Ampere (A100, RTX 3090)
```

## Usage
```bash
./mc_sim -i 10000 -t 1E-3 -f -v            # 10k paths, first-hit, verbose
./mc_sim -i 10000 -c -f -v                 # CPU vs GPU comparison
./mc_sim -i 5000 -w -r 8E-6 -e             # with walls, record everything
./mc_sim_cpu -i 10000 -f -l 3E-7 -t 1E-2   # CPU-only, 1D limit test
```

## Validation
```bash
# Quick pre-commit check
./scripts/validate_before_commit.sh

# Manual validation against analytical solution
./mc_sim_cpu -i 10000 -f -l 3E-7 -t 1E-2
scripts/.venv/bin/python scripts/validate_1d_firsthit.py build/output_h.csv \
    --dist 3E-7 --vel 1E-4 --timestep 1E-7 --timestop 1E-2
```

## Known Issues
- Data race on shared `x_new`/`y_new` device pointers in wall reflection
- Per-timestep cudaMemcpy bottleneck limits GPU speedup
- RNG re-initialized per kernel call (clock64 seed) — correlated streams possible
