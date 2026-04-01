# Molecular Communication GPU Simulation -- Domain Reference

This document summarizes the physics model, key literature, validation approaches,
and known limitations relevant to the GPU-based molecular communication particle
simulator developed in the Cain thesis and related work by Rogers et al.

---

## 1. Physics Model and Equations

### 1.1 Governing Equation -- Forward Kolmogorov / Fokker-Planck

The probability density function (pdf) for a messenger molecule's position is
governed by the forward Kolmogorov equation (also called the Fokker-Planck
equation):

```
d p_B(t, xi; xi_0, t_0)       1   3   3   d^2 D_{j,k} p_B         3   d (v_j p_B)
------------------------  =  --- SUM SUM  ----------------  -  SUM  ------------
         d t                  2  j=1 k=1  d xi_j  d xi_k       j=1    d xi_j
```

where:
- `D_{j,k}` is the diffusion constant (set to `2 * D_B` when j = k, zero otherwise
  for anisotropic diffusion)
- `v_j` is the drift velocity in direction j
- `p_B(t, xi; xi_0, t_0)` is the pdf for emission location xi at time t with
  starting point xi_0 and time t_0

For non-interacting, sufficiently small BA emissions in an aqueous medium with
timescales much larger than the relaxation time, the solution is a multivariate
normal distribution:

```
p_B(t, xi; xi_0, t_0) = 1 / ((2*pi)^(3/2) * |Sigma|^(1/2))
                          * exp(-0.5 * (xi - mu)^T * Sigma^{-1} * (xi - mu))
```

with mean vector `mu = v_x * (t - t_0) * [1, 0, 0]^T + xi_0` and covariance
`Sigma = 2 * D_B * (t - t_0) * I`.

### 1.2 Brownian Motion -- Discrete Simulation

Each molecule is advanced per time step using the Euler method:

```
x_next += sqrt(2 * D_B * dt) * randn_x
y_next += sqrt(2 * D_B * dt) * randn_y
z_next += sqrt(2 * D_B * dt) * randn_z
```

where `D_B` is the diffusion coefficient (m^2/s), `dt` is the time step (s), and
`randn` is a standard normal random variable.

### 1.3 Laminar Drift

When drift (laminar flow) is present, a deterministic velocity component is added
to the drift direction (z in the thesis convention):

```
z_next += sqrt(2 * D_B * dt) * randn_z + velocity * dt
```

The drift is assumed to be a single-layer laminar flow acting in one direction.
In a real blood vessel, the Hagen-Poiseuille parabolic velocity profile applies:

```
v(r) = (1 / (4*mu)) * (dP / L) * (R^2 - r^2)
```

but for simulation purposes in the cell-free layer, a constant velocity
approximation is used.

### 1.4 Diffusion Coefficient -- Stokes-Einstein Relation

The diffusion coefficient is related to physical parameters via:

```
D = k_B * T / (6 * pi * eta * R_s)
```

where `k_B = 1.38e-23 J/K` is the Boltzmann constant, `T` is temperature (K),
`eta` is viscosity (Pa*s), and `R_s` is the molecule radius. A typical value used
in the simulations is `D_B = 1e-11 m^2/s`, corresponding to a messenger molecule
radius of approximately 1 nm.

### 1.5 Hit Detection

#### 1.5.1 Spherical Receiver (3D, Diffusion Only)

The reaction rate at a perfectly absorbing spherical receiver of radius `a` at
distance `d` from the transmitter (from Schulten & Kosztin, "Lectures in
Theoretical Biophysics", Eq. 3.108):

```
H_Diff(t | d, t_0) = (a / d) * (1 / sqrt(4*pi*D*(t-t_0))) * ((d-a) / (t-t_0))
                      * exp(-(d-a)^2 / (4*D*(t-t_0)))
```

The asymptotic hit probability for a perfect absorber as `(t - t_0) -> inf`:

```
Pr(m_k hit R) = a / d
```

This is the upper bound on diffusion-only channel capacity: `C_Diff <= a/d`.

The cumulative detection probability as a function of time (from Schulten,
Eq. 3.116):

```
N_Detect(t | d, t_0) = (a*(alpha-1)) / (d*alpha)
  * (1 + erfc[(a-d) / sqrt(4*D*(t-t_0))]
  - exp[alpha*(d-a) + alpha^2*D*(t-t_0)]
  * erfc[(d - a + 2*D*(t-t_0)*alpha) / sqrt(4*D*(t-t_0))])
```

When `alpha -> inf` (perfectly absorbing):

```
lim N_Detect = (a/d) * erfc((d-a) / sqrt(4*D*(t-t_0)))
```

#### 1.5.2 Planar Receiver (1D, Drift + Diffusion)

The first-passage time pdf for a 1D Brownian motion with drift follows an inverse
Gaussian distribution (from Kadloor et al. 2012, Eq. 12):

```
p_passage(t) = (b / sqrt(4*pi*D_B*t^3)) * exp(-(v*t - b)^2 / (4*D_B*t))
```

where `b` is the distance from transmitter to receiver and `v` is the drift
velocity. When drift is present and positive, `Pr(m_k hit R) = 1` in 1D.

This is also an inverse Gaussian distribution with:
- mean `mu = b / v`
- shape parameter `lambda = b^2 / (2*D)`

### 1.6 Wall Reflection

Blood vessel walls are modeled as a perfect cylinder of radius `R`. When a
molecule's position after a time step places it outside the cylinder, it is
reflected back inside. The thesis validation (`TobyThesisTest_walls.m`) sets up
parameters including:

- `vesselRadius_m = 8e-6` (8 micrometers, typical capillary radius)
- `transmitterRadius_m = vesselRadius_m - 200e-9` (200 nm from the wall)

The reflection keeps particles confined to the cell-free layer near the vessel
wall.

### 1.7 Peclet Number

The Peclet number relates drift velocity to diffusion:

```
Pe_x = v_x * L / D_B
```

where `L` is a characteristic distance. It indicates whether transport is
dominated by advection (high Pe) or diffusion (low Pe).

---

## 2. Key Papers and Their Contributions

### 2.1 Core Thesis

**"Molecular Communication GPU Simulation"** -- Toby Cain (thesis, advised by
Dr. Yun Tian and Dr. Uri Rogers, Eastern Washington University)

- Applies GPU (CUDA) parallel processing to molecular communication particle
  simulation
- Implements Brownian motion + laminar drift on the GPU using the Euler method
- Key design: calculates all N paths at one time step simultaneously (SIMD),
  rather than each path sequentially, to exploit GPU parallelism
- Validates against analytical solutions for both diffusion-only (3D spherical
  receiver) and drift+diffusion (1D planar receiver)
- Implements hit detection (spherical receiver, planar limit), first-hit time
  recording, and cylindrical vessel wall reflection
- Uses cuRAND for GPU random number generation; CPU version uses Marsaglia
  polar method
- Speed comparison: GPU up to 3.44x faster than single-core CPU for 10,000
  paths with first-hit recording; advantage diminishes beyond ~1E5 paths due
  to GPU-to-host memory transfer bottleneck
- Data output: CSV format with (x, y, z, time_step) per hit event

### 2.2 Molecular Communication Channel Theory

**Rogers & Koh, "Exploring Molecular Distributed Detection"** (conference paper)

- Introduces Molecular Distributed Detection (MDD) concept
- Derives sensor observation probability from the Fokker-Planck equation
- Defines the MDD system: Biological Agent, N sensors, MC channel, fusion center
- Applies Chair-Varshney optimal fusion rule and counting rule for hypothesis
  testing
- Uses ROC curve analysis for detection performance
- Directly relevant: provides the Fokker-Planck equation (Eq. 1-2) and the
  multivariate normal solution (Eq. 3) used throughout the simulation

**Rogers & Michaelis, "Channel Capacity for Molecular Communications Under
Brownian Motion"** (ICC 2018 submission)

- Establishes MC channel capacity bounds for both diffusion-only and
  drift+diffusion channels in R^3
- Models MC as a deletion channel (or erasure channel with timing synchronization)
- Proves `C <= Pr(hit) = a/d` for diffusion-only channel (binary erasure channel)
- Derives `H_Diff(t|d,t_0)` (Eq. 9) -- the key analytical first-hitting time
  distribution used for validation
- Derives cumulative detection probability `N_Detect(t|d,t_0)` (Eq. 10-11)
- Shows that multiple absorbing receivers in proximity reduce each other's
  channel capacity
- Uses the vector Euler algorithm for stochastic simulation with 500,000 sample
  paths
- Introduces passage time formula (Eq. 18) for drift+diffusion in R^1
- Simulation parameters: `D = 1e-11 m^2/s`, `a = 10 nm`, `v_x = 1 mm/s`,
  capillary diameter 8 um

**Kadloor, Adve, & Eckford, "Molecular Communication Using Brownian Motion
with Drift"** (IEEE Trans. NanoBioscience, 2012)

- Foundational paper for Brownian motion with drift channel model
- Derives the diffusion equation with drift (Eq. 5-7):
  `d/dt P_X = (D * d^2/dx^2 + v * d/dx) P_X`
- Solution: `P_X(x,t) = 1/sqrt(4*pi*D*t) * exp(-(x - v*t)^2 / (4*D*t))`
- Derives first-passage time distribution using method of images (Eq. 10-12):
  `f(t) = zeta / sqrt(4*pi*D*t^3) * exp(-(v*t - zeta)^2 / (4*D*t))`
- Calculates mutual information for single and two-molecule pulse-position
  modulation
- Shows optimal degree distributions for transmitter input

**Kim, Eckford, & Chae, "Symbol Interval Optimization for Molecular
Communication with Drift"** (IEEE Trans. NanoBioscience, 2014)

- Proposes symbol interval optimization to balance ISI and data rate
- Uses the inverse Gaussian distribution for first-hitting time (Eq. 1-2)
- Defines hitting probability CDF using standard Gaussian CDF
- Applies IMoSK (isomer-based molecule shift keying) modulation
- Blood vessel parameters: capillaries with diameter 8 um, velocity 7.9e2 um/s

### 2.3 Channel Characterization and Interference

**Noel, Cheung, & Schober, "Optimal Receiver Design for Diffusive Molecular
Communication With Flow and Additive Noise"** (IEEE Trans. NanoBioscience, 2014)

- Derives optimal and suboptimal receiver detectors for 3D diffusive MC with flow
- Uses Stokes-Einstein relation for diffusion coefficient (Eq. 4)
- Models received signal as Poisson random variable
- Incorporates enzyme degradation (Michaelis-Menten kinetics) for ISI mitigation
- Derives concentration at receiver including flow in arbitrary direction
  (effective distance formula)
- Relevant for future work on receiver-side detection algorithms

**Pierobon & Akyildiz, "Intersymbol and Co-channel Interference in
Diffusion-based Molecular Communication"** (ICC Workshop, 2012)

- Analyzes ISI and co-channel interference (CCI) in diffusion-based MC
- Uses Green's function: `g(x,t) = 1/sqrt((4*pi*D*t)^3) * exp(-|x|^2/(4*D*t))`
- Derives closed-form ISI and CCI formulas using diffusion wave theory
- Shows attenuation `alpha(w) = sqrt(w/(2D))` and phase velocity `v_p = sqrt(2Dw)`

**Bicen & Akyildiz, "End-to-End Propagation Noise and Memory Analysis for
Molecular Communication over Microfluidic Channels"** (IEEE Trans. Comm., 2014)

- Develops end-to-end channel model for flow-induced MC over microfluidic channels
- Analyzes molecular memory effect from inter-diffusion of previously transmitted
  signals
- Shows AWGN model validity for certain microfluidic channel configurations
- Uses Taylor dispersion for effective diffusion coefficient in rectangular channels

### 2.4 Blood Vessel and Biological Models

**Felicetti et al., "Modeling CD40-Based Molecular Communications in Blood
Vessels"** (IEEE Trans. NanoBioscience, 2014)

- Models MC between platelets and endothelial cells via CD40 signaling
- Hagen-Poiseuille flow profile: `v(r) = (1/(4*mu)) * (dP/L) * (R^2 - r^2)`
- Stokes drag force: `F_d = 6*pi*mu*R*v_p`
- Advection-diffusion equation: `dC/dt + v . grad(C) = D_m * laplacian(C)`
- Simulation parameters (Table I): vessel diameter 60 um, mean velocity 0.5 mm/s,
  blood viscosity 0.0013 Pa*s, temperature 310 K, RBC radius 3.5 um,
  simulation time step 100 us
- Uses BiNS simulator (Java-based) with inelastic collisions between particles

**Nakano, Moore, Wei, Vasilakos, & Shuai, "Molecular Communication and
Networking: Opportunities and Challenges"** (IEEE Trans. NanoBioscience, 2012)

- Comprehensive survey of MC architecture, applications, and physical models
- Describes three propagation classes: random walk, random walk with drift,
  random walk with reaction
- Application areas: drug delivery, health monitoring, lab-on-chip,
  environmental monitoring
- Identifies key challenges: scalable networking protocols, reliable
  communication over stochastic channels

### 2.5 GPU Computing

**Owens et al., "GPU Computing"** (Proceedings of the IEEE, 2008)

- Foundational survey of GPGPU computing
- Describes GPU architecture: streaming multiprocessors, SIMD execution,
  throughput-oriented design
- CPU faster for < 10^4 elements; GPU surpasses for larger parallel workloads
- Applications in computational biophysics: protein folding, molecular dynamics

**Januszewski & Kostur, "Accelerating numerical solution of Stochastic
Differential Equations with CUDA"** (arXiv, 2009)

- Directly relevant GPU/CUDA approach for SDE simulation
- Presents Algorithm 1: CUDA kernel for advancing Brownian particle by `m * dt`
  time steps -- the same pattern used in the thesis
- Uses per-thread RNG state (xor-shift), Box-Muller transform for Gaussian
  variates
- Reports up to 675x speedup over CPU
- Key design: each path in a separate thread, multiple time steps per kernel
  invocation, parameters cached in local variables
- Uses Stochastic Runge-Kutta 2nd order (SRK2) integrator
- Finding: 10^5 independent realizations sufficient for convergence

---

## 3. Validation Approaches and Analytical Solutions

### 3.1 Test 1: 3D Diffusion-Only First-Hit Time (TobyThesisTest.m)

**Setup:**
- Point source transmitter at origin
- Spherical receiver of radius `a = 10 nm` at distance `d = 50 nm`
- Diffusion coefficient `D_B = 1e-11 m^2/s`
- Time step `dt = 1e-7 s`, simulation stop time `t_stop = 1e-3 s`
- No drift (`v = 0`)

**Analytical reference** (Eq. 3.108 from Schulten, "Lectures in Theoretical
Biophysics"):

```
H_Diff(t|d,t_0) = (a/d) * (1/sqrt(4*pi*D*(t-t_0))) * ((d-a)/(t-t_0))
                   * exp(-(d-a)^2 / (4*D*(t-t_0)))
```

**Hit probability** (Eq. 3.116 from Schulten):

```
Pr(m_k hit R) = (a/d) * erfc((d-a) / sqrt(4*D*t_stop))
```

**Validation method:** GPU simulation first-hit times are histogrammed (120 bins),
scaled by the analytical hit probability, and overlaid against the analytical pdf.
The thesis shows excellent agreement (Figure 4.1).

### 3.2 Test 2: 1D Drift+Diffusion First-Passage Time (TobyThesisTest_1_D_FirstHit.m)

**Setup:**
- Point source at origin, planar receiver at distance `b` in z-direction
- Diffusion `D_B = 1e-11 m^2/s`, drift velocity `v = 1e-4 m/s`
- Distance `b = 3e-7 m` (300 nm)
- Time step `dt = 1e-7 s`, stop time `t_stop = 1e-2 s`

**Analytical reference** (inverse Gaussian distribution):

```
p_passage(t) = (b / sqrt(4*pi*D_B*t^3)) * exp(-(v*t - b)^2 / (4*D_B*t))
```

**Validation method:** GPU hit times histogrammed and compared to analytical pdf.
Since `Pr(hit) = 1` in 1D with positive drift, the histogram is normalized
directly as `f / (sum(f) * deltaT)`. The thesis shows good agreement
(Figure 4.2).

### 3.3 Test 3: Distance Variant with Drift (TobyThesisTest_dist.m)

Same as Test 2 but with different distance/velocity parameters and using 10,000
particles. Uses the same inverse Gaussian analytical solution. Hit probability is
empirically estimated as the fraction of particles that reached the receiver.

### 3.4 Test 4: Vessel Wall Validation (TobyThesisTest_walls.m)

**Setup:**
- Cylindrical blood vessel with radius `R = 8e-6 m` (8 um)
- Transmitter at `(0, R - 200nm, 0)` -- in the cell-free layer, 200 nm from wall
- Spherical receiver of radius `a = 10 nm` at `(0, R-200nm, 50nm)` -- 50 nm
  downstream in z
- 120,000 particles

**Analytical reference:** Same diffusion-only H_Diff formula, using the 3D
Euclidean distance from transmitter to receiver center. This validates that the
wall reflection mechanism does not corrupt the diffusion statistics for particles
that remain within the interior.

### 3.5 Random Number Generation Validation

Both C/CPU (Marsaglia polar method) and CUDA (cuRAND built-in) random number
generators were validated against MATLAB's `randn` output using cross-correlation
analysis. The cross-correlation peaks near the center were small, confirming
minimal correlation (Figures 3.1, 3.2 in thesis).

---

## 4. Simulation Parameters -- Standard Values

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Diffusion coefficient | D_B | 1e-11 | m^2/s |
| Blood mean velocity | v_x | 1e-3 (1) | mm/s |
| Capillary diameter | L | 8 (range 6-12) | um |
| Receiver radius | a | 10 | nm |
| Transmitter-receiver distance | d | 50 | nm |
| Time step | dt | 1e-7 | s |
| Simulation stop time | t_stop | 1e-3 | s |
| Temperature | T | 310 | K |
| Blood viscosity | eta | 0.0013 | Pa*s |
| Messenger molecule radius | R_s | ~1 | nm |

---

## 5. GPU Implementation Design

### 5.1 Parallelization Strategy

- Each GPU thread handles one particle path for a single time step
- All N particle positions at time `t_n` are computed in parallel, then `t_{n+1}`,
  etc. (step-synchronous)
- This differs from CPU approach where each path is computed to completion
  sequentially
- Thread index: `idx = blockIdx.x * blockDim.x + threadIdx.x`
- Parameters (D_B, velocity, positions) loaded into local registers at kernel
  start
- RNG seed is per-thread, maintained across kernel invocations

### 5.2 Data Storage

- Full path output: (x, y, z) at every time step -- produces very large files
  (8+ GB)
- Hit-time output: only records (x, y, z, time_step) when particle enters the
  receiver volume
- First-hit output: records only the first time a particle reaches the receiver
- Collision detection is performed in-kernel at each time step (real-time
  checking rather than post-processing)

### 5.3 Performance Characteristics

- GPU advantage appears at ~5,000+ paths for first-hit recording
- Peak speedup ~3.44x at 10,000 paths (GTX 1070 vs single-core i7-7700K)
- Speedup decreases beyond ~1E5 paths due to per-step GPU-to-host memory copy
  and CSV write overhead
- Memory transfer is the primary bottleneck, not computation
- Adding drift has negligible impact on computation time (one extra multiply-add)

---

## 6. Known Limitations and Future Work Directions

### 6.1 From the Thesis

1. **Memory transfer bottleneck:** The GPU kernel is called from a CPU loop for
   each time step, requiring GPU-to-host memory copies. Computing entire paths
   on-GPU without intermediate copies would improve performance.

2. **Red blood cell (RBC) interactions:** RBCs are not modeled. Their tumbling
   motion in the core region creates turbulent flow that pushes smaller particles
   to the cell-free layer. Modeling RBC collisions would increase realism.

3. **Molecule-molecule interactions:** Simulated molecules are independent
   (non-interacting). Future work could add inter-particle collisions or
   chemical reactions.

4. **Additional obstacles:** Only cylindrical walls and spherical receivers are
   implemented. More complex geometries (bifurcations, irregular vessel shapes,
   additional obstructions) are identified as future work.

5. **Parabolic flow profile:** Only constant-velocity (single-layer) laminar flow
   is implemented. The full Hagen-Poiseuille parabolic profile, where velocity
   depends on radial position, is not yet modeled.

### 6.2 From the Literature

1. **No closed-form 3D drift+diffusion hit probability:** The analytical
   first-passage time solution for drift+diffusion exists only in R^1. In R^3
   with drift, `Pr(m_k hit R)` must be estimated numerically (Rogers & Michaelis
   2018). The passage time formula from R^1, when scaled by the hit probability,
   does approximate the R^3 drift+diffusion hitting rate, but this requires
   further study.

2. **Channel memory and ISI:** The diffusion process creates inherent
   inter-symbol interference because previously transmitted molecules can arrive
   in later symbol intervals. The inverse Gaussian distribution has a long tail
   at low velocities, making ISI a serious impairment (Kim et al. 2014).

3. **Multiple receiver interference:** When multiple receivers are in proximity,
   absorbing receivers shadow each other, reducing channel capacity by factors
   of 2-3x or more (Rogers & Michaelis 2018).

4. **Receiver detection probability:** The thesis assumes perfect absorption
   (`Pr(detect | hit) = 1`). In reality, receivers have finite reaction rates
   controlled by the parameter `omega` in the reaction rate equation (Eq. 8 in
   Rogers & Michaelis). Imperfect detection further reduces capacity.

5. **Enzyme degradation:** Noel et al. (2014) model enzyme molecules that degrade
   information molecules via Michaelis-Menten kinetics, which can mitigate ISI
   but is not implemented in the current simulator.

6. **Concentration-based signaling:** The current simulator tracks individual
   particles. An alternative approach models concentration fields directly using
   the advection-diffusion PDE, which may be more efficient for certain
   scenarios.

7. **Capacity bounds:** Theoretical capacity is bounded by `C <= Pr(hit)` for
   deletion channels. Tighter bounds using Martin kernel capacity
   (`Cap_M(R)`) exist but are complex to compute (Rogers & Michaelis 2018,
   Eq. 15-16).

---

## 7. Key References (Organized by Topic)

### Molecular Communication Foundations
- Nakano et al., "Molecular Communication and Networking," IEEE Trans.
  NanoBioscience, 2012
- Nakano, Eckford, & Haraguchi, "Molecular Communication," Cambridge Univ.
  Press, 2013

### Brownian Motion and Diffusion Theory
- Karatzas & Shreve, "Brownian Motion and Stochastic Calculus," 2nd ed.,
  Springer, 1991
- Gardiner, "Handbook of Stochastic Methods for Physics, Chemistry and the
  Natural Sciences," Springer, 2004
- Schulten & Kosztin, "Lectures in Theoretical Biophysics," Univ. of Illinois,
  2000 (primary source for analytical validation equations)
- Dhont, "An Introduction to Dynamics of Colloids," Elsevier, 1996

### Channel Modeling and Capacity
- Kadloor, Adve, & Eckford, "MC Using Brownian Motion with Drift," IEEE Trans.
  NanoBioscience, 2012
- Rogers & Michaelis, "Channel Capacity for MC Under Brownian Motion," ICC 2018
- Rogers & Koh, "Exploring Molecular Distributed Detection," conference paper
- Pierobon & Akyildiz, "ISI and Co-channel Interference in Diffusion-based MC,"
  ICC 2012

### Receiver Design and Modulation
- Noel, Cheung, & Schober, "Optimal Receiver Design for Diffusive MC," IEEE
  Trans. NanoBioscience, 2014
- Kim, Eckford, & Chae, "Symbol Interval Optimization for MC with Drift," IEEE
  Trans. NanoBioscience, 2014
- Kabir et al., "D-MoSK Modulation," 2015

### Blood Vessel Environment
- Felicetti et al., "Modeling CD40-Based MC in Blood Vessels," IEEE Trans.
  NanoBioscience, 2014
- Mayrovitz et al., "Blood Velocity Measurement," Cardiovascular Diseases, 1981
- Koutsiaris et al., "Blood Velocity Pulse Quantification," Microvascular
  Research, 2010

### GPU Computing
- Owens et al., "GPU Computing," Proceedings of the IEEE, 2008
- Januszewski & Kostur, "Accelerating Numerical Solution of SDEs with CUDA,"
  2009
- Lee et al., "Debunking the 100x GPU vs. CPU Myth," 2010

### Noise and Interference
- Bicen & Akyildiz, "End-to-End Propagation Noise and Memory for MC over
  Microfluidic Channels," IEEE Trans. Comm., 2014
- Noel, Cheung, & Schober, "A Unifying Model for External Noise Sources," IEEE
  JSAC, 2014
- Pierobon & Akyildiz, "Capacity of Diffusion-based MC with Memory," 2013
