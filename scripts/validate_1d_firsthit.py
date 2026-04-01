#!/usr/bin/env python3
"""
Validate first-hitting time distribution against analytical solution.

Ported from TobyThesisTest_1_D_FirstHit.m (written by Dr. Uri Rogers).
Compares simulation CSV output against the 1-D inverse Gaussian passage
time distribution (thesis eq 4.3, Kadloor et al. 2012 Eq. 12):

    p_passage(t) = b / sqrt(4*pi*Db*t^3) * exp(-(v*t - b)^2 / (4*Db*t))

Validation uses one-sample KS test (formal pass/fail) with NRMSE as
supplementary metric. See docs/validation_methodology.md for rationale.

Usage:
    python validate_1d_firsthit.py <csv_file> [--no-plot]
"""

import argparse
import sys
import numpy as np


def load_hit_times(filename, time_step=1E-7):
    """Load CSV and extract hit times from the timestep column.

    Handles both output formats:
      CPU: x, y, z, iteration, timestep   (timestep is last real column)
      GPU: x, y, z, timestep              (timestep is last real column)
    Trailing commas in CSV produce NaN columns — we find the last
    non-NaN column per row as the timestep.
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    if data.size == 0:
        return np.array([])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Filter out fully-NaN rows (e.g. from header lines or blank lines)
    data = data[~np.isnan(data).all(axis=1)]
    if data.size == 0:
        return np.array([])
    # Timestep is always the last non-NaN column in each row
    # (works for both CPU 5-col and GPU 4-col formats)
    n_rows = data.shape[0]
    timesteps = np.full(n_rows, np.nan)
    for i in range(n_rows):
        row = data[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            timesteps[i] = valid[-1]
    hit_times = timesteps[~np.isnan(timesteps)] * time_step
    hit_times = hit_times[hit_times > 0]
    return hit_times


def analytical_passage_time(t, b, Db, v):
    """1-D first passage time inverse Gaussian PDF (thesis eq 4.3)."""
    return b / np.sqrt(4 * np.pi * Db * t**3) * np.exp(-(v * t - b)**2 / (4 * Db * t))


def ks_test_invgauss(hit_times, b, Db, v):
    """
    One-sample KS test against the inverse Gaussian distribution.

    scipy.stats.invgauss parameterization:
      mu_scipy = 2*D / (b*v)     (shape parameter)
      scale_scipy = b^2 / (2*D)  (scale parameter)

    See docs/validation_methodology.md for derivation.
    """
    from scipy.stats import kstest, invgauss

    mu_scipy = 2 * Db / (b * v)
    scale_scipy = b**2 / (2 * Db)
    frozen = invgauss(mu_scipy, loc=0, scale=scale_scipy)

    stat, pvalue = kstest(hit_times, frozen.cdf)
    return stat, pvalue


def nrmse_histogram(hit_times, b, Db, v, num_bins=120):
    """Supplementary NRMSE on histogram (for human readability, not pass/fail)."""
    time_scale = 1E3  # msec

    counts, bin_edges = np.histogram(hit_times * time_scale, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    delta_t = (bin_centers[1] - bin_centers[0]) / time_scale
    pdf_sim = counts / (np.sum(counts) * delta_t)

    bin_centers_sec = bin_centers / time_scale
    pdf_analytical_at_bins = analytical_passage_time(bin_centers_sec, b, Db, v)

    mask = counts > 0
    if np.sum(mask) == 0:
        return float('inf'), None, None

    peak = np.max(pdf_analytical_at_bins[mask])
    rmse = np.sqrt(np.mean((pdf_sim[mask] - pdf_analytical_at_bins[mask])**2))
    nrmse = rmse / peak if peak > 0 else float('inf')

    return nrmse, pdf_sim, bin_centers


def main():
    parser = argparse.ArgumentParser(description='Validate 1-D first-hit times against analytical solution')
    parser.add_argument('csv_file', help='Path to simulation output CSV')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting, just print stats')
    parser.add_argument('--dist', type=float, default=3E-7, help='Distance to limit (m), default 3E-7')
    parser.add_argument('--db', type=float, default=1E-11, help='Diffusion coefficient (m^2/s)')
    parser.add_argument('--vel', type=float, default=1E-4, help='Drift velocity (m/s)')
    parser.add_argument('--timestep', type=float, default=1E-7, help='Simulation time step (s)')
    parser.add_argument('--timestop', type=float, default=1E-2, help='Simulation stop time (s)')
    parser.add_argument('--alpha', type=float, default=0.001, help='KS test significance level (default 0.001)')
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    hit_times = load_hit_times(args.csv_file, args.timestep)
    if len(hit_times) == 0:
        print("ERROR: No valid hits loaded.")
        sys.exit(1)

    print(f"  {len(hit_times)} hits loaded")
    print(f"  Hit time range: {hit_times.min()*1E3:.4f} - {hit_times.max()*1E3:.4f} ms")

    print(f"\nParameters:")
    print(f"  Distance (b): {args.dist}")
    print(f"  Db: {args.db}")
    print(f"  Velocity: {args.vel}")
    print(f"  Time step: {args.timestep}")

    # Formal test: one-sample KS against inverse Gaussian
    print(f"\n=== KS Test (alpha={args.alpha}) ===")
    ks_stat, ks_pvalue = ks_test_invgauss(hit_times, args.dist, args.db, args.vel)
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {ks_pvalue:.6f}")
    ks_pass = ks_pvalue > args.alpha
    print(f"  {'PASS' if ks_pass else 'FAIL'}")

    # Supplementary: NRMSE on histogram
    nrmse, pdf_sim, bin_centers = nrmse_histogram(hit_times, args.dist, args.db, args.vel)
    print(f"\n=== NRMSE (supplementary) ===")
    print(f"  Normalized RMSE: {nrmse:.4f}")

    # Overall
    print(f"\n{'PASS' if ks_pass else 'FAIL'} - 1D first-hit validation (KS p={ks_pvalue:.4f})")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        time_scale = 1E3
        tt = np.linspace(1E-8, args.timestop, 10000)
        pdf_analytical = analytical_passage_time(tt, args.dist, args.db, args.vel)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PDF comparison
        if pdf_sim is not None and bin_centers is not None:
            delta_t = (bin_centers[1] - bin_centers[0]) / time_scale
            axes[0].plot(bin_centers, pdf_sim, 'k', linewidth=2, label='Simulation')
        axes[0].plot(tt * time_scale, pdf_analytical, 'r--', linewidth=2, label='Analytical')
        axes[0].set_xlabel('time (msec)', fontsize=14)
        axes[0].set_ylabel('p_passage(t)', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].set_title(f'PDF Comparison (NRMSE={nrmse:.4f})')

        # CDF comparison (what KS test actually compares)
        from scipy.stats import invgauss
        mu_scipy = 2 * args.db / (args.dist * args.vel)
        scale_scipy = args.dist**2 / (2 * args.db)
        frozen = invgauss(mu_scipy, loc=0, scale=scale_scipy)

        sorted_times = np.sort(hit_times)
        ecdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        axes[1].plot(sorted_times * time_scale, ecdf, 'k', linewidth=2, label='Empirical CDF')
        t_cdf = np.linspace(sorted_times.min(), sorted_times.max(), 10000)
        axes[1].plot(t_cdf * time_scale, frozen.cdf(t_cdf), 'r--', linewidth=2, label='Analytical CDF')
        axes[1].set_xlabel('time (msec)', fontsize=14)
        axes[1].set_ylabel('CDF', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].set_title(f'CDF Comparison (KS p={ks_pvalue:.4f})')

        plt.tight_layout()
        plt.savefig('validation_1d_firsthit.png', dpi=150)
        print("\nPlot saved to validation_1d_firsthit.png")
        plt.show()

    return 0 if ks_pass else 1


if __name__ == '__main__':
    sys.exit(main())
