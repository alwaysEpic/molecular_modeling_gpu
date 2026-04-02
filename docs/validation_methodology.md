# Validation Methodology

Scientific rationale for the simulation validation approach.

## Statistical Tests

### Why KS Test Over NRMSE

NRMSE on histograms is bin-dependent (change binning, change result), has no principled
acceptance threshold, and is not a formal statistical test. The Kolmogorov-Smirnov test
compares empirical CDF against theoretical CDF with no binning, has a known null
distribution, provides a rigorous p-value, and is consistent (detects any departure given
enough samples).

We keep NRMSE as a human-readable supplementary metric but gate pass/fail on KS p-value.

References:
- NIST Engineering Statistics Handbook: chi-squared is "generally inferior" to KS/AD
- Law (2015) "Simulation Modeling and Analysis" Ch. 6: KS test for output validation
- Devroye (1986) "Non-Uniform Random Variate Generation" Ch. 2: goodness-of-fit testing

### Significance Level: alpha = 0.001 for CI

At alpha = 0.05, ~5% of CI runs would false-fail. At alpha = 0.001, false positive rate
is 0.1%. With n=10,000, KS can still detect CDF differences of ~0.019, which catches any
real implementation bug.

### Sample Size

| n | KS critical (alpha=0.001) | Detectable CDF shift |
|---|---------------------------|---------------------|
| 1,000 | 0.062 | ~0.06 |
| 10,000 | 0.019 | ~0.02 |
| 100,000 | 0.006 | ~0.006 (may detect discretization) |

1,000 paths is adequate for a quick smoke test (pre-commit). At 10,000 paths the KS
test becomes sensitive enough to detect the O(sqrt(dt)) discrete boundary-crossing
bias (Gobet 2000), which causes legitimate KS failures — see Known Limitations below.

### Minimum Sample Size for Valid Tests

KS test is mathematically valid at any n but has no statistical power below ~20
samples (Conover 1999, D'Agostino & Stephens 1986). Tests are skipped with a
message when n < 20. Results flagged as low-power when 20 <= n < 50.

Binomial test (scipy.stats.binomtest) is exact at any n >= 1, but skipped
below n=20 for consistency.

## Test 1: 1D First-Passage Time with Drift

### Analytical Solution
Inverse Gaussian distribution (Kadloor et al. 2012, Eq. 12):

    p_passage(t) = b / sqrt(4*pi*D*t^3) * exp(-(v*t - b)^2 / (4*D*t))

Note: Thesis eq 4.3 has a typesetting error (`-(vt - b^2)` instead of `-(v*t - b)^2`).
Code is correct.

### scipy Parameterization
    mu_scipy = 2*D / (b*v)       # shape parameter
    scale_scipy = b^2 / (2*D)    # scale parameter
    frozen = invgauss(mu_scipy, loc=0, scale=scale_scipy)
    stat, pvalue = kstest(samples, frozen.cdf)

### Parameters (matching MATLAB TobyThesisTest_1_D_FirstHit.m)
- b = 3E-7 m (distance)
- D = 1E-11 m^2/s
- v = 1E-4 m/s
- dt = 1E-7 s
- t_stop = 1E-2 s

## Test 2: 3D Diffusion-Only First-Hitting Time

### Analytical Solution (Schulten eq 3.108)

    H_Diff(t|d,t0) = (a/d) * 1/sqrt(4*pi*D*(t-t0)) * (d-a)/(t-t0)
                      * exp(-(d-a)^2 / (4*D*(t-t0)))

This integrates to Pr(hit) = (a/d) * erfc((d-a)/sqrt(4*D*t_stop)), not to 1.

### Two Separate Tests Required

1. **Hit probability (binomial test):** Is the observed fraction of hits consistent
   with the theoretical Pr(hit)?
   
       binomtest(n_hit, n_total, p=Pr_hit_analytical)

2. **Hit-time distribution (KS test):** Do the observed hitting times match the
   *conditional* distribution (H_Diff normalized to integrate to 1)?
   
   CDF computed by numerical integration of H_Diff on a fine grid, then normalized
   so CDF(t_max) = 1.

This separation is standard practice in the literature (Noel et al. 2014, Yilmaz et al.
2014).

### Parameters (matching MATLAB TobyThesisTest.m)
- a = 10E-9 m (receiver radius)
- d = 50E-9 m (distance)
- D = 1E-11 m^2/s
- dt = 1E-7 s
- t_stop = 1E-3 s
- No drift (v = 0)

## Test 3: CPU/GPU Agreement

Two-sample KS test comparing hit-time distributions from both backends.
Different RNG implementations (stdlib rand vs curand Philox), so outputs are
not identical — we test statistical consistency.

- Hit rate comparison: within expected binomial variance
- Two-sample KS: p > 0.01

## Test 4: RNG Quality

- Statistical moments (mean, std, kurtosis) against N(0,1) with 3-sigma bounds
- Autocorrelation at lags 1-50 against 95% white-noise bound
- Anderson-Darling normality test

These are application-level checks. NVIDIA's cuRAND is validated against TestU01
BigCrush (106 tests). Our tests verify the integration is correct, not the generator
itself.

## Histogram NRMSE (supplementary)

Kept for human-readable output and backward compatibility with thesis validation
approach. Not used as pass/fail criterion. Only displayed when >= 1000 hits are
available — below this threshold the histogram is too sparse for NRMSE to be
informative (dominated by shot noise, not systematic error).

Theoretical basis: histogram MISE converges as O(n^{-2/3}) for optimal bin width
(Wasserman 2004, "All of Statistics", Theorem 20.9). This gives RMSE ~ O(n^{-1/3}).
For NRMSE < 10%, need n >= 1000. For heavy-tailed distributions like the inverse
Gaussian, n >= 5000 is preferred (Silverman 1986, Section 2.5). The chi-squared
"5 expected events per bin" rule (Cochran 1954) with 120 bins requires n >= 600.

Normalization:
- 1D (Pr(hit)=1): counts / (sum(counts) * deltaT)
- 3D (Pr(hit)<1): counts * (prob_hit / trapz(counts, t_seconds))

## Known Limitations

### Discrete Boundary-Crossing Bias

The Euler-Maruyama integrator is exact for our constant-coefficient SDE, but hit
detection is discrete: a particle can cross through the receiver between timesteps
without being detected. This introduces a systematic negative bias in hit probability
and a slight shift in the hit-time distribution, scaling as O(sqrt(dt)) (Gobet 2000,
Math. Comp. 69:225-259).

At 10,000 paths with dt=1E-7, the KS test is powerful enough to detect this bias
(KS stat ~0.024, p ~5E-5 against the continuous analytical solution). This is a
real physical limitation of discrete-time simulation, not a code bug.

**Impact (without correction):**
- 1D first-hit KS test fails at n >= 10,000 with alpha=0.001
- 3D binomial test fails (observed ~14% hit rate vs theoretical ~15.5%)
- Both pass comfortably at n=1,000 (insufficient power to detect the bias)

**Fix (implemented):** Brownian bridge correction — at each timestep, compute the
probability that the continuous path crossed the boundary between x_n and x_{n+1}
using P(cross) = exp(-(c-a)*(c-b) / (D*dt)), and sample accordingly.
This eliminates the dominant discretization error without reducing dt.

Implemented in both long and wide GPU kernels and the CPU path. After correction,
1D KS and 3D binomial tests pass at 10k+ paths. See `docs/brownian_bridge.md`
for full derivation and references.
