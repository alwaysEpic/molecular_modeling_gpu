# Molecular Communication GPU Simulator

## Project Overview
GPU-accelerated Monte Carlo simulation of molecular communication in blood vessels.
Simulates particle diffusion via Brownian motion with optional laminar drift, modeling
how nanoscale messenger molecules move through the bloodstream.

Based on a master's thesis applying CUDA parallelism to molecular communication research.

## Architecture
- **src/main.cu** — Core CUDA simulation. GPU kernel (`d_update`) runs one thread per
  particle path, computing one timestep at a time across all paths in parallel.
  The CPU calls the kernel in a loop over timesteps, with cudaMemcpy each step for
  collision detection on the host side.
- **src/timing.c/h** — Wall-clock timing utility using gettimeofday.
- **matlab/** — Reference MATLAB Fokker-Planck implementations used for validation.
- **docs/thesis.pdf** — Full thesis document with equations, validation, and benchmarks.

## Physics Model
- Brownian motion: `x += sqrt(2 * Db * deltaT) * randn` for each axis
- Laminar drift: `z += velocity * deltaT` (z-direction only)
- Wall reflection: parametric line-circle intersection in cylindrical blood vessel
- Collision detection: spherical receiver (radius `rec_rad` at distance `rec_distance`)
  or 1D planar limit

## Key Parameters
- `Db` = 1E-11 m^2/s (diffusion coefficient)
- `velocity` = 1E-4 m/s (laminar flow)
- `deltaT` = 1E-7 s (time step)
- `radius` = 8E-6 m (blood vessel radius)
- GPU RNG: curandStatePhilox4_32_10_t

## Build
```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75   # adjust for your GPU
make
```

## CLI Options
```
./mc_sim -i 10000 -t 1E-3 -f -v          # 10k paths, first-hit, verbose
./mc_sim -i 5000 -w -r 8E-6 -e           # with walls, record everything
./mc_sim -i 10000 -c -f -n               # CPU comparison, no drift
```

## Known Issues / Future Work
- GPU speedup degrades at high iteration counts due to per-timestep cudaMemcpy
- Wall reflection has debug prints and incomplete multi-bounce handling
- RNG initialized per-kernel-call (clock64 seed) — could be improved with persistent state
- Architecture target needs updating for modern GPUs
- Consider Rust port for non-GPU portions
