# Brownian Bridge Boundary Crossing Correction

Comprehensive reference for implementing the Brownian bridge correction
in the molecular communication particle simulator.

## The Problem

Our simulator advances particles in discrete timesteps using Euler-Maruyama:

    X_{n+1} = X_n + sqrt(2*D*dt) * Z_n + v*dt    (Z_n ~ N(0,1))

Between steps n and n+1, the continuous path is unobserved. A particle can
cross through a boundary (receiver or wall) and return to the same side —
completely undetected at either endpoint. This causes:

1. **Underestimation of hit probability** — observed ~14% vs theoretical ~15.5%
   for our 3D test, ~97.8% vs 100% for 1D with drift
2. **Bias in first-passage time distribution** — KS test fails at 10k paths
   (p ~5E-5 at alpha=0.001)
3. **Error scaling as O(sqrt(dt))** — Gobet (2000), Math. Comp. 69:225-259

Neither the thesis nor any of its reference papers discuss this issue. All
analytical formulas (Schulten Eq. 3.108, Kadloor Eq. 12) assume continuous
observation. The discretization bias was identified during project optimization
and is documented in validation_methodology.md.

## The Brownian Bridge

### Definition

Given X(t_n) = a and X(t_{n+1}) = b, the Brownian bridge is the conditional
distribution of the continuous path between these known endpoints:

    X(t) | {X(t_n)=a, X(t_{n+1})=b} ~ N(mu(t), sigma^2(t))

where, with s = t - t_n and s_bar = dt - s:

    mu(t)      = a + (s/dt) * (b - a)         (linear interpolation)
    sigma^2(t) = 2*D*s*s_bar/dt               (parabolic, max at midpoint)

The mean is the straight line between endpoints. The variance is zero at
both endpoints and maximal at the midpoint (sigma^2_max = D*dt/2).

### Boundary Crossing Probability — The Core Formula

For a particle at X_n = a and X_{n+1} = b, both on the same side of a
planar boundary at level c, the probability that the continuous path
crossed c during (t_n, t_{n+1}) is:

    P_cross = exp(-(c-a)*(c-b) / (D*dt))

This is derived from the **reflection principle** (Bachelier 1900, Lévy 1939):
paths from a to b that touch c are in bijection with paths from a to (2c-b)
via the strong Markov property of Brownian motion. The ratio of transition
densities gives the exponential.

### Key Properties

1. **Exact for a single planar boundary** with constant diffusion D
2. **Drift-independent**: the bridge between known endpoints has the same
   distribution regardless of drift (Revuz & Yor, Prop. IX.3.8). The drift
   determines the endpoints; the bridge between them is drift-free. Therefore
   **use the same formula with or without drift**.
3. **When both endpoints are on opposite sides** of c, P_cross = 1 (the
   continuous path must have crossed — already handled by endpoint check)
4. **Argument is always negative** (a,b same side of c) so P_cross in (0,1)

### Derivation Sketch

For standard Brownian motion W(t) from 0 to b over [0,T], touching level c > 0:

    P(max W >= c | W(T)=b) = p(2c-b, T) / p(b, T)
                            = exp(-(2c-b)^2/(2T)) / exp(-b^2/(2T))
                            = exp(-2c(c-b)/T)

Shifting start to a and rescaling for diffusion D:

    P_cross = exp(-2*(c-a)/(sqrt(2D)) * (c-b)/(sqrt(2D)) / dt)
            = exp(-(c-a)(c-b)/(D*dt))

Full derivation with completing-the-square for the bridge distribution is in
Karatzas & Shreve (1991), Chapter 3.

## 3D Spherical Boundary Extension

For a spherical receiver of radius r_s, use the **tangent-plane approximation**
(Northrup, Allison, McCammon 1984; Lamm & Schulten 1983):

    d_n   = r_n - r_s      (distance from sphere surface at step n)
    d_n+1 = r_n+1 - r_s    (distance at step n+1)

    P_cross = exp(-d_n * d_n+1 / (D*dt))

This treats the sphere locally as a plane, which is valid when the diffusion
step sqrt(2*D*dt) is small compared to the sphere radius r_s. For our
parameters: sqrt(2*1E-11*1E-7) = 4.5E-9 m, r_s = 1E-8 m — ratio ~0.45,
so the approximation is reasonable but not exact.

The exact formula for a 3D Bessel process involves infinite series of Bessel
functions (Pitman & Yor 1982). The leading correction is a factor r_s/r_n
which matters when the particle is far from the sphere. For our use case
(particles near the receiver), the NAM approximation is standard.

## Reflecting (Wall) Boundaries

For reflecting boundaries (vessel walls):

- **Explicit crossing** (endpoint past wall): already handled by d_reflection
- **Bridge crossing** (both endpoints inside, path touched wall): the bridge
  correction detects the wall hit, but since the wall is reflecting (not
  absorbing), the endpoint position X_{n+1} is unchanged — the path hit
  the wall and bounced back to X_{n+1}
- **Error is O(dt)** not O(sqrt(dt)) for reflecting walls, because the
  curvature correction is higher-order. Lower priority than receiver correction.

Wall bridge correction matters when:
- Counting wall hits (e.g., for surface reactions)
- Partially absorbing walls
- The wall is very close (within a diffusion step)

## Implementation in Our Code

### Boundary Check Locations (6 total)

| Location | File | Type | Bridge Priority |
|----------|------|------|----------------|
| 1D planar limit | d_simulate_isolated, line ~159 | Absorbing | HIGH |
| Spherical receiver | d_simulate_isolated, line ~164 | Absorbing | HIGH |
| Wall reflection | d_simulate_isolated, line ~150 | Reflecting | LOW |
| 1D planar limit | d_update, line ~100 | Absorbing | HIGH |
| Spherical receiver | d_update, line ~110 | Absorbing | HIGH |
| Wall reflection | d_update, line ~84 | Reflecting | LOW |

Same checks also exist in simulation_cpu.cpp (lines 82-86).

### Algorithm for Each Step

    1. Compute new position (Euler-Maruyama step)
    2. Save old position (pz_old for limit, r_old for receiver)
    3. Check explicit crossing (new position past boundary)
       → If yes: record hit, done
    4. If both endpoints on same side of boundary:
       Compute P_cross = exp(-(c-a)*(c-b) / (D*dt))
       Draw U ~ Uniform(0,1)
       If U < P_cross: record hit (bridge-detected crossing)

### RNG for the Uniform Draw

We already call curand_normal4(&state) per step, producing 4 normal values
but only using 3 (rn.x, rn.y, rn.z). The 4th value (rn.w) can be
transformed to U(0,1) via the normal CDF:

    u = 0.5f * (1.0f + erff(rn.w * 0.70710678f))

This avoids any extra RNG calls. erff() maps to a hardware instruction
(MUFU) on CUDA, so it is essentially free. expf() for the crossing
probability also maps to hardware (MUFU.EX2).

### Crossing Time Estimation

When a bridge crossing is detected, we need a crossing time for the
first-passage record. Options (increasing accuracy/cost):

1. **Midpoint**: t_cross = t_n + dt/2. Crude, O(dt) accurate.
2. **Linear interpolation**: t_cross = t_n + dt * d_n/(d_n + d_n+1).
   Better for explicit crossings.
3. **Inverse Gaussian sample**: exact conditional distribution of first
   crossing time for the bridge. Most accurate but expensive (rejection
   sampling needed).

Recommendation: use midpoint for now (simplest, sufficient for histogram
validation). The O(dt) error in crossing time is same order as the
remaining discretization error after bridge correction.

## Error Analysis

| Error Source | Without Bridge | With Bridge |
|-------------|---------------|-------------|
| Missed boundary crossings | O(sqrt(dt)) | **Eliminated** |
| SDE discretization | O(dt) | O(dt) |
| Sphere curvature approx | N/A | O(dt * D/r_s^2) |
| Multiple crossings/step | N/A | O(dt^2) |
| **Overall** | **O(sqrt(dt))** | **O(dt)** |

The bridge correction eliminates the dominant error term, promoting
convergence from half-order to first-order. To halve the error:
- Without bridge: need dt/4 (quadruple computation)
- With bridge: need dt/2 (double computation)

## Multiple Boundaries

When both receiver and wall are present:

1. Check absorbing boundaries first (receiver takes priority)
2. Apply bridge correction to each boundary independently
3. Error from independence assumption: O(dt^{3/2}) — negligible when
   boundaries are well-separated relative to sqrt(2*D*dt)

For our parameters, the receiver (50nm from origin) and wall (8um radius)
are far apart relative to the diffusion step (~4.5nm), so independent
treatment is valid.

## Validation Tests

### Test 1: 1D KS Test Should Pass at 10k Paths

Currently fails with KS stat ~0.024, p ~5E-5. After bridge correction,
the simulation should match the continuous-time inverse Gaussian, and
this test should PASS.

### Test 2: 3D Binomial Hit Rate Should Match Theory

Currently ~14% vs theoretical 15.5%. After bridge correction, the
observed hit rate should converge to the theoretical value within
binomial sampling variance.

### Test 3: Convergence Rate Test (new)

Run the simulation at dt and dt/2. The difference in hit probability
should scale as O(dt) with bridge (vs O(sqrt(dt)) without). This
directly validates that the correction promotes the convergence order.

### Test 4: Bridge Probability Unit Test (new)

Generate known (a, b) pairs with a known boundary c and D*dt. Compute
P_cross analytically and verify the simulation's bridge acceptance rate
matches over many samples.

## Key References

- Gobet (2000), "Weak approximation of killed diffusion using Euler
  schemes", Stoch. Proc. Appl. 87:167-197
- Northrup, Allison, McCammon (1984), "Brownian dynamics simulation of
  diffusion-influenced bimolecular reactions", J. Chem. Phys. 80:1517-1524
- Lamm & Schulten (1983), "Extended Brownian dynamics II", J. Chem. Phys.
  78:2713-2734
- Andrews & Bray (2004), "Stochastic simulation of chemical reactions
  with spatial resolution", Physical Biology 1:137-151
- Karatzas & Shreve (1991), Brownian Motion and Stochastic Calculus,
  2nd ed., Springer, Chapter 3
- Revuz & Yor (2005), Continuous Martingales and Brownian Motion, 3rd ed.,
  Springer, Prop. IX.3.8
- Opplestrup et al. (2006), "First-passage Monte Carlo algorithm",
  Phys. Rev. Lett. 97:230602
