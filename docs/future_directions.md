# Future Directions

Possible research extensions for the molecular communication GPU simulator.
Ranked by estimated impact and implementation complexity.

---

## 1. Enzyme Degradation for ISI Mitigation

**Impact:** High | **Complexity:** Low

Inter-symbol interference (ISI) is the dominant impairment in diffusion-based
MC. Molecules from previous symbols arrive late and corrupt detection. Enzyme
degradation destroys lingering molecules and is the standard mitigation approach.

At low substrate concentration (typical for MC), Michaelis-Menten kinetics
reduce to first-order decay. In simulation, each molecule is independently
removed per timestep with probability:

    P_deg = 1 - exp(-k_deg * dt)

One uniform random draw per molecule per step. Works in both long and
wide kernels with no architectural changes.

**Publishable result:** BER vs degradation rate curves for OOK modulation,
validating particle-level degradation against the analytical model in
Adam Noel, Karen Cheung, and Robert Schober (2014).

**References:**
- Adam Noel, Karen Cheung, Robert Schober, "Optimal Receiver Design
  for Diffusive MC With Flow and Additive Noise," IEEE Trans.
  NanoBioscience, Vol. 13, No. 3, 2014
- Massimiliano Pierobon, Ian Akyildiz, "Intersymbol and Co-channel
  Interference in Diffusion-based MC," Int'l Workshop on Molecular
  and Nanoscale Commun. (ICC), 2012

## 2. Multiple Molecule Emissions (OOK Modulation)

**Impact:** High | **Complexity:** Low

Required for communication performance analysis. The transmitter emits N
molecules per symbol (OOK: bit-1 = emit N, bit-0 = emit 0). The receiver
counts arrivals within the symbol period and compares to a detection threshold.
BER is the primary metric.

Typical values from the literature: N = 100–10,000 molecules per symbol.
Emitted molecules are non-interacting (physically justified — see note below).

**Implementation:** Multiple sequential simulation runs with different
emission times, accumulate arrivals per symbol period, compute BER.

## 3. Red Blood Cell (RBC) Obstacles

**Impact:** High | **Complexity:** High

RBCs occupy ~45% of blood volume. Their presence fundamentally alters
molecular transport. The thesis identified this as a primary future work item.

RBCs can be modeled as mobile rigid spheres (radius ~3.5um) undergoing
Brownian motion with Stokes drag. Messenger molecules are point particles
that elastically reflect off RBC surfaces.

**Implementation requires:**
- Cell list spatial data structure (sorting-based with Thrust)
- Neighbor-finding kernel (27-cell stencil in Cartesian grid)
- Elastic reflection off moving spherical obstacles
- Uses the d_update per-step kernel path (positions in global memory)

**References:**
- Luca Felicetti, Mauro Femminella, Gianluca Reali, et al., "Modeling
  CD40-Based MC in Blood Vessels," IEEE Trans. NanoBioscience, Vol. 13,
  No. 3, 2014
- Toby Cain, "Molecular Communication GPU Simulation," Thesis, Eastern
  Washington University, 2018 (Section 6: Future Work)

## 4. Parabolic Flow Profile

**Impact:** Medium | **Complexity:** Low

Replace constant-velocity drift with the Hagen-Poiseuille profile:

    v(r) = v_max * (1 - r²/R²)

Particles near the vessel center travel faster than those near the wall.
Affects hit probability and ISI for receivers at different radial positions.

**Implementation:** Replace constant `driftStep` with position-dependent
calculation in the kernel. Minimal code change.

## 5. Partially Absorbing Receiver

**Impact:** Medium | **Complexity:** Low

Perfect absorption (current) overestimates channel capacity. Real receivers
have finite reaction rates. When a molecule reaches the receiver, absorb with
probability P_abs < 1, otherwise reflect.

**References:**
- Uri Rogers, Matthew Michaelis, "Channel Capacity for Molecular
  Communications Under Brownian Motion," ICC 2018
- Radek Erban, S. Jonathan Chapman, "Reactive Boundary Conditions for
  Stochastic Simulations of Reaction-Diffusion Processes," Physical
  Biology, Vol. 4, No. 1, 2007

## 6. Multiple Receivers

**Impact:** Medium | **Complexity:** Medium

Multiple spherical receivers at different positions. Enables study of spatial
dependency and receiver shadowing effects.

Uri Rogers and Min-Sung Koh (ICASSP 2017) showed non-linear spatial
dependency with multiple receivers. Uri Rogers and Matthew Michaelis
(ICC 2018) showed absorbing receivers shadow each other, reducing
channel capacity 2–3x.

---

## Relationship to AcCoRD

AcCoRD (Noel et al., Nano Comm. Networks 11:44-75, 2017) is the closest
thing to a standard MC simulator. It is C-based, CPU-only, single-threaded,
and essentially unmaintained since 2019. Its microscopic diffusion model uses
the same Euler-Maruyama step as ours, but wraps it in an event-driven
priority queue with linked-list molecule storage — architecturally
incompatible with GPU parallelism.

Rather than porting AcCoRD to GPU (fighting its serial event-driven core),
the approach is to bring AcCoRD-equivalent physics into our GPU-native
architecture. Several AcCoRD features map directly onto embarrassingly
parallel per-particle operations:

| AcCoRD Feature | GPU Difficulty | Notes |
|----------------|---------------|-------|
| First-order degradation | Low | One random draw per molecule per step (section 1 above) |
| Multiple molecule types | Low | Separate arrays or type index per particle |
| Partial absorption | Low | Probability check at hit detection (section 5 above) |
| Parabolic flow | Low | Position-dependent drift, one multiply (section 4 above) |
| Reversible binding | Medium | Receiver state requires atomics |
| Second-order reactions | Hard | Neighbor finding, spatial hashing (section 3 above) |
| Mesoscopic RDME | Not viable | Gillespie algorithm is fundamentally serial |

Features 1-4 above cover the common AcCoRD use cases without touching
the architecture. This would position the simulator as a GPU-accelerated
alternative to AcCoRD for the subset of MC problems that are
particle-based and embarrassingly parallel — which is most of them.

No published GPU-accelerated full MC simulator exists. The only GPU work
in MC is Stroobant and Felicetti et al. (Nano Comm. Networks, 2019) who
GPU-accelerated collision detection only, reporting ~100x speedup on
that component alone.

---

## Note: Molecule-Molecule Collisions

Molecule-molecule collisions are physically negligible at MC concentrations
and should not be prioritized:

- Volume fraction ~10⁻⁸ (orders of magnitude below any interaction threshold)
- Mean free path between messenger molecules (~40mm) exceeds the channel
  length (~100um) by a factor of 400
- All standard MC simulators (Smoldyn, BiNS, N3Sim) neglect them
- No published MC paper has found them to be significant

The relevant form of crowding is background macromolecular crowding
(cellular proteins at 20–40% volume fraction causing anomalous subdiffusion),
which is a different physical effect.

---

## Current Capabilities

| Feature | Status |
|---------|--------|
| Brownian motion + drift | Done (1,400x speedup vs thesis) |
| Spherical receiver | Done (KS-validated) |
| 1D planar limit | Done (KS-validated at 10M paths) |
| Wall reflection | Done (bug-fixed, validated) |
| Brownian bridge correction | Done (3D hit rate fixed) |
| Truncated CDF validation | Done (accounts for censored tails) |
| Long kernel | Done (positions in registers) |
| CPU reference path | Done (independent validation) |
| Reproducibility (--seed) | Done |
| 10M particle scale | Done (30s on T4) |
