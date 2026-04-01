#!/usr/bin/env python3
"""
Validate 3D diffusion-only first-hit time against analytical solution.

Ported from TobyThesisTest.m (written by Dr. Uri Rogers).
Compares simulation CSV output against the 3D spherical receiver
hit probability and conditional hit-time distribution.

Two separate tests (see docs/validation_methodology.md):
  1. Binomial test: is the observed hit fraction consistent with Pr(hit)?
  2. KS test: do hitting times match the conditional H_Diff distribution?

Analytical solution (Schulten eq 3.108):
    H_Diff(t|d,t0) = (a/d) * 1/sqrt(4*pi*D*(t-t0)) * (d-a)/(t-t0)
                      * exp(-(d-a)^2 / (4*D*(t-t0)))

Usage:
    python validate_3d_diffusion.py <csv_file> [--no-plot] [--total-paths N]
"""

import argparse
import sys
import numpy as np


def load_hit_times(filename, time_step=1E-7):
    """Load CSV and extract hit times from the timestep column."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Filter out any NaN rows
    data = data[~np.isnan(data).all(axis=1)]
    valid_cols = np.sum(~np.isnan(data[0, :]))
    if valid_cols >= 5:
        timesteps = data[:, 4]  # CPU output
    else:
        timesteps = data[:, 3]  # GPU output
    hit_times = timesteps[~np.isnan(timesteps)] * time_step
    hit_times = hit_times[hit_times > 0]
    return hit_times


def h_diff_pdf(t, a, d, Db):
    """3D diffusion-only first-hitting time PDF (Schulten eq 3.108, t0=0)."""
    result = np.zeros_like(t, dtype=float)
    mask = t > 0
    result[mask] = (a / d) * (1.0 / np.sqrt(4 * np.pi * Db * t[mask])) * \
                   ((d - a) / t[mask]) * np.exp(-(d - a)**2 / (4 * Db * t[mask]))
    return result


def analytical_hit_probability(a, d, Db, t_stop):
    """Finite-time hit probability (Schulten eq 3.116, perfectly absorbing limit)."""
    from scipy.special import erfc
    return (a / d) * erfc((d - a) / np.sqrt(4 * Db * t_stop))


def build_conditional_cdf(a, d, Db, t_stop):
    """
    Build the conditional CDF of H_Diff (normalized to integrate to 1).

    The raw H_Diff integrates to Pr(hit) < 1. Since our samples are only
    from particles that DID hit, we must compare against the conditional
    distribution. See docs/validation_methodology.md.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d

    # Fine grid — characteristic time scale is (d-a)^2 / (2*D)
    t_grid = np.linspace(1e-12, t_stop, 500000)
    pdf_vals = h_diff_pdf(t_grid, a, d, Db)

    cdf_raw = cumulative_trapezoid(pdf_vals, t_grid, initial=0)

    # Normalize to 1 — this converts to the conditional distribution
    if cdf_raw[-1] > 0:
        cdf_conditional = cdf_raw / cdf_raw[-1]
    else:
        cdf_conditional = cdf_raw

    cdf_func = interp1d(t_grid, cdf_conditional,
                        bounds_error=False,
                        fill_value=(0.0, 1.0),
                        kind='linear')
    return cdf_func


def test_hit_probability(n_hit, n_total, a, d, Db, t_stop):
    """Binomial test: is observed hit fraction consistent with theory?"""
    from scipy.stats import binomtest

    p_theory = analytical_hit_probability(a, d, Db, t_stop)
    result = binomtest(n_hit, n_total, p=p_theory, alternative='two-sided')
    return p_theory, result.pvalue


def test_hit_distribution(hit_times, a, d, Db, t_stop, alpha=0.001):
    """KS test: do hitting times match the conditional H_Diff distribution?"""
    from scipy.stats import kstest

    cdf_func = build_conditional_cdf(a, d, Db, t_stop)
    stat, pvalue = kstest(hit_times, cdf_func)
    return stat, pvalue


def nrmse_histogram(hit_times, a, d, Db, t_stop, num_bins=120):
    """Supplementary NRMSE on histogram (for human readability, not pass/fail)."""
    time_scale = 1E3
    p_hit = analytical_hit_probability(a, d, Db, t_stop)

    counts, bin_edges = np.histogram(hit_times * time_scale, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize using trapezoid rule, scaled by hit probability (matches MATLAB)
    trap_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    trap_val = trap_func(counts, bin_centers / time_scale)
    if trap_val > 0:
        pdf_sim = counts * (p_hit / trap_val)
    else:
        return float('inf'), None, None

    bin_centers_sec = bin_centers / time_scale
    pdf_analytical_at_bins = h_diff_pdf(np.array(bin_centers_sec), a, d, Db)

    mask = counts > 0
    if np.sum(mask) == 0:
        return float('inf'), None, None

    peak = np.max(pdf_analytical_at_bins[mask])
    rmse = np.sqrt(np.mean((pdf_sim[mask] - pdf_analytical_at_bins[mask])**2))
    nrmse = rmse / peak if peak > 0 else float('inf')

    return nrmse, pdf_sim, bin_centers


def main():
    parser = argparse.ArgumentParser(description='Validate 3D diffusion-only first-hit times')
    parser.add_argument('csv_file', help='Path to simulation output CSV')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--receiver-radius', type=float, default=10E-9, help='Receiver radius (m), default 10nm')
    parser.add_argument('--distance', type=float, default=50E-9, help='Transmitter-receiver distance (m), default 50nm')
    parser.add_argument('--db', type=float, default=1E-11, help='Diffusion coefficient (m^2/s)')
    parser.add_argument('--timestep', type=float, default=1E-7, help='Simulation time step (s)')
    parser.add_argument('--timestop', type=float, default=1E-3, help='Simulation stop time (s)')
    parser.add_argument('--total-paths', type=int, default=None, help='Total paths simulated (for hit rate test)')
    parser.add_argument('--alpha', type=float, default=0.001, help='KS test significance level (default 0.001)')
    args = parser.parse_args()

    a = args.receiver_radius
    d = args.distance

    print(f"Loading {args.csv_file}...")
    hit_times = load_hit_times(args.csv_file, args.timestep)
    if len(hit_times) == 0:
        print("ERROR: No valid hits loaded.")
        sys.exit(1)

    print(f"  {len(hit_times)} hits loaded")
    print(f"  Hit time range: {hit_times.min()*1E3:.4f} - {hit_times.max()*1E3:.4f} ms")

    print(f"\nParameters:")
    print(f"  Receiver radius (a): {a}")
    print(f"  Distance (d): {d}")
    print(f"  Db: {args.db}")
    print(f"  Theoretical Pr(hit) as t->inf: a/d = {a/d:.4f}")

    all_pass = True

    # Test 1: Hit probability (binomial test)
    if args.total_paths:
        print(f"\n=== Test 1: Hit Probability (binomial) ===")
        p_theory, binom_pvalue = test_hit_probability(
            len(hit_times), args.total_paths, a, d, args.db, args.timestop)
        empirical_rate = len(hit_times) / args.total_paths
        print(f"  Theoretical Pr(hit, t_stop={args.timestop}): {p_theory:.4f}")
        print(f"  Empirical hit rate: {empirical_rate:.4f} ({len(hit_times)}/{args.total_paths})")
        print(f"  Binomial test p-value: {binom_pvalue:.6f}")
        binom_pass = binom_pvalue > args.alpha
        print(f"  {'PASS' if binom_pass else 'FAIL'}")
        if not binom_pass:
            all_pass = False
    else:
        print(f"\n  (Skipping hit probability test — provide --total-paths to enable)")

    # Test 2: Hit-time distribution (KS test against conditional CDF)
    print(f"\n=== Test 2: Hit-Time Distribution (KS, alpha={args.alpha}) ===")
    ks_stat, ks_pvalue = test_hit_distribution(hit_times, a, d, args.db, args.timestop, args.alpha)
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {ks_pvalue:.6f}")
    ks_pass = ks_pvalue > args.alpha
    print(f"  {'PASS' if ks_pass else 'FAIL'}")
    if not ks_pass:
        all_pass = False

    # Supplementary: NRMSE
    nrmse, pdf_sim, bin_centers = nrmse_histogram(hit_times, a, d, args.db, args.timestop)
    print(f"\n=== NRMSE (supplementary) ===")
    print(f"  Normalized RMSE: {nrmse:.4f}")

    print(f"\n{'PASS' if all_pass else 'FAIL'} - 3D diffusion validation (KS p={ks_pvalue:.4f})")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        from scipy.stats import kstest

        time_scale = 1E3
        tt = np.linspace(1E-8, args.timestop, 10000)
        pdf_analytical = h_diff_pdf(tt, a, d, args.db)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PDF comparison
        if pdf_sim is not None and bin_centers is not None:
            axes[0].plot(bin_centers, pdf_sim, 'k', linewidth=2, label='Simulation')
        axes[0].plot(tt * time_scale, pdf_analytical, 'r--', linewidth=2, label='Analytical')
        axes[0].set_xlabel('time (msec)', fontsize=14)
        axes[0].set_ylabel('$H_{Diff}(t|d,t_0)$', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].set_title(f'PDF Comparison (NRMSE={nrmse:.4f})')

        # CDF comparison
        cdf_func = build_conditional_cdf(a, d, args.db, args.timestop)
        sorted_times = np.sort(hit_times)
        ecdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        axes[1].plot(sorted_times * time_scale, ecdf, 'k', linewidth=2, label='Empirical CDF')
        t_cdf = np.linspace(sorted_times.min(), sorted_times.max(), 10000)
        axes[1].plot(t_cdf * time_scale, cdf_func(t_cdf), 'r--', linewidth=2, label='Conditional CDF')
        axes[1].set_xlabel('time (msec)', fontsize=14)
        axes[1].set_ylabel('CDF', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].set_title(f'CDF Comparison (KS p={ks_pvalue:.4f})')

        plt.tight_layout()
        plt.savefig('validation_3d_diffusion.png', dpi=150)
        print("\nPlot saved to validation_3d_diffusion.png")
        plt.show()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
