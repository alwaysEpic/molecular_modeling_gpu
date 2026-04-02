# Molecular Communication GPU Simulator

GPU-accelerated Monte Carlo simulation of molecular communication in blood vessels.
Simulates particle diffusion via Brownian motion with optional laminar drift,
modeling how nanoscale messenger molecules move through the bloodstream.

Based on the master's thesis *"The Application of GPU to Molecular Communication
Studies"* (Cain, Eastern Washington University, 2018).

## Quick Start (Google Colab)

The easiest way to run the simulator is through Google Colab, which provides
free access to NVIDIA GPUs. No local GPU or CUDA installation required.

### 1. Create a GitHub Personal Access Token

The repo is private, so Colab needs a token to clone it.

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token** > **Generate new token (classic)**
3. Give it a name (e.g. "colab")
4. Set expiration (90 days is fine)
5. Check the **repo** scope (full control of private repositories)
6. Click **Generate token**
7. **Copy the token** — you won't be able to see it again

### 2. Open the Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
   - Sign in with a Google account if you don't have one
2. Click **File > Open notebook**
3. Select the **GitHub** tab
4. Paste the repo URL: `https://github.com/alwaysEpic/molecular_modeling_gpu`
   - You may need to check "Include private repos" and authorize Colab
5. Select `scripts/colab_build_test.ipynb`

### 3. Add Your Token to Colab Secrets

1. In the left sidebar, click the **key icon** (Secrets)
2. Click **Add a new secret**
3. Name: `GITHUB_PAT`
4. Value: paste the token you copied in step 1
5. Toggle **Notebook access** on

### 4. Select a GPU Runtime

1. Go to **Runtime > Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU** (free tier) or **A100** (if available)
3. Click **Save**

### 5. Run the Notebook

Click **Runtime > Run all** or step through cells one at a time.

The notebook is organized as a progression:

| Section | What it does | Time |
|---------|-------------|------|
| Setup | Builds the simulator | ~1 min |
| Particle Path | 3D random walk visualization | seconds |
| Validation | KS tests against analytical solutions (1D, 3D, walls) | ~1 min |
| Consistency | CPU/GPU agreement, reproducibility, RNG quality | ~2 min |
| Performance | Thesis vs current speedup comparison with charts | ~5 min |
| Stress Tests | 1M and 10M path validation with plots | ~2 min |

Total run time is about 10-15 minutes on a T4.

## Local Build (requires CUDA)

```bash
git clone https://github.com/alwaysEpic/molecular_modeling_gpu.git
cd molecular_modeling_gpu
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This builds:
- `mc_sim` — GPU simulator (requires NVIDIA GPU + CUDA toolkit)
- `mc_sim_cpu` — CPU-only reference (always builds)

### Usage

```bash
# 10k particles, first-hit mode, verbose
./mc_sim -i 10000 -f -v

# 1D planar receiver with drift
./mc_sim -i 10000 -f -l 3E-7 -t 1E-2

# Force wide kernel (for development)
./mc_sim -i 10000 -f -l 3E-7 -t 1E-2 -W

# CPU-only
./mc_sim_cpu -i 10000 -f -l 3E-7 -t 1E-2
```

### Validation

```bash
# Run local pre-commit test suite (CPU-only, no GPU needed)
./scripts/validate_before_commit.sh
```

## GPU Kernel Architectures

**Wide kernel** (`d_update`): One kernel launch per timestep. All particles
advance together, positions in global memory. Required for future particle
interactions.

**Long kernel** (`d_simulate_isolated`): Single launch, each thread runs one
particle's full path with positions in registers. Much faster for independent
particles (first-hit mode).

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-i N` | 1000 | Number of particle paths |
| `-t T` | 1E-3 | Simulation duration (seconds) |
| `-d DT` | 1E-7 | Time step (seconds) |
| `-f` | off | First-hit recording mode |
| `-l D` | off | 1D planar limit distance (meters) |
| `-n` | off | Disable drift |
| `-w` | off | Enable vessel walls |
| `-W` | off | Force wide kernel |
| `-S N` | 0 | RNG seed (0 = time-based) |
| `-v` | off | Verbose output |
| `-h` | — | Show all options |

## Project Structure

```
src/
  common/
    params.h              # SimParams struct
    cli.h                 # CLI parsing
  main.cu                 # GPU entry point
  main_cpu.cpp            # CPU-only entry point
  simulation_cpu.cpp/.h   # CPU reference implementation
  simulation_gpu.cu/.h    # GPU kernels (long + wide)
scripts/
  colab_build_test.ipynb  # Colab notebook (start here)
  validate_*.py           # Validation scripts
  validate_before_commit.sh
docs/
  thesis.pdf              # Original thesis
  brownian_bridge.md      # Bridge correction derivation
  domain_reference.md     # Physics and literature reference
  future_directions.md    # Research directions
```
