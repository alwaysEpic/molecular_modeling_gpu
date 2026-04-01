#!/usr/bin/env python3
"""
Validate 3D diffusion in cylindrical blood vessel with wall reflection.

Ported from TobyThesisTest_walls.m (written by Dr. Uri Rogers).
Tests that wall reflection does not corrupt the diffusion statistics
by comparing hit-time distribution against the free-space H_Diff
analytical solution.

The analytical solution assumes free-space diffusion. Agreement here
means reflections are not introducing artifacts (valid when the receiver
is far from the wall relative to the diffusion length scale).

Setup (matching MATLAB):
  - Cylindrical vessel radius: 8um
  - Transmitter at (0, R-200nm, 0) — in the cell-free layer
  - Receiver at (0, R-200nm, 50nm) — 50nm downstream in z
  - 3D Euclidean distance: 50nm (same x,y; 50nm apart in z)
  - Diffusion only (no drift)

Usage:
    python validate_3d_walls.py <csv_file> [--no-plot] [--total-paths N]
"""

import argparse
import sys
import math
import numpy as np


def load_hit_times(filename, time_step=1E-7):
    """Load CSV and extract hit times (last non-NaN column per row)."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    if data.size == 0:
        return np.array([])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[~np.isnan(data).all(axis=1)]
    if data.size == 0:
        return np.array([])
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


def h_diff_pdf(t, a, d, Db):
    """3D diffusion-only first-hitting time PDF (Schulten eq 3.108, t0=0)."""
    result = np.zeros_like(t, dtype=float)
    mask = t > 0
    result[mask] = (a / d) * (1.0 / np.sqrt(4 * np.pi * Db * t[mask])) * \
                   ((d - a) / t[mask]) * np.exp(-(d - a)**2 / (4 * Db * t[mask]))
    return result


def analytical_hit_probability(a, d, Db, t_stop):
    """Finite-time hit probability (Schulten eq 3.116)."""
    from scipy.special import erfc
    return (a / d) * erfc((d - a) / np.sqrt(4 * Db * t_stop))


def build_conditional_cdf(a, d, Db, t_stop):
    """Build conditional CDF of H_Diff normalized to integrate to 1."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d

    t_grid = np.linspace(1e-12, t_stop, 500000)
    pdf_vals = h_diff_pdf(t_grid, a, d, Db)
    cdf_raw = cumulative_trapezoid(pdf_vals, t_grid, initial=0)
    if cdf_raw[-1] > 0:
        cdf_conditional = cdf_raw / cdf_raw[-1]
    else:
        cdf_conditional = cdf_raw
    return interp1d(t_grid, cdf_conditional, bounds_error=False,
                    fill_value=(0.0, 1.0), kind='linear')


def main():
    parser = argparse.ArgumentParser(description='Validate 3D diffusion with wall reflection')
    parser.add_argument('csv_file', help='Path to simulation output CSV')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--total-paths', type=int, default=None, help='Total paths simulated')
    parser.add_argument('--alpha', type=float, default=0.001, help='KS test significance level')

    # Matching TobyThesisTest_walls.m parameters
    parser.add_argument('--vessel-radius', type=float, default=8E-6, help='Blood vessel radius (m)')
    parser.add_argument('--wall-offset', type=float, default=200E-9, help='Distance from wall to transmitter (m)')
    parser.add_argument('--receiver-radius', type=float, default=10E-9, help='Receiver radius (m)')
    parser.add_argument('--receiver-offset-z', type=float, default=50E-9, help='Receiver z offset from transmitter (m)')
    parser.add_argument('--db', type=float, default=1E-11, help='Diffusion coefficient (m^2/s)')
    parser.add_argument('--timestep', type=float, default=1E-7, help='Simulation time step (s)')
    parser.add_argument('--timestop', type=float, default=0.4E-3, help='Simulation stop time (s)')
    args = parser.parse_args()

    # Compute 3D Euclidean distance from transmitter to receiver center
    # Transmitter: (0, R - wall_offset, 0)
    # Receiver:    (0, R - wall_offset, receiver_offset_z)
    # Distance:    receiver_offset_z (since x,y are the same)
    tx_x, tx_y, tx_z = 0.0, args.vessel_radius - args.wall_offset, 0.0
    rx_x, rx_y, rx_z = 0.0, args.vessel_radius - args.wall_offset, args.receiver_offset_z
    d = math.sqrt((tx_x - rx_x)**2 + (tx_y - rx_y)**2 + (tx_z - rx_z)**2)
    a = args.receiver_radius

    print(f"Loading {args.csv_file}...")
    hit_times = load_hit_times(args.csv_file, args.timestep)
    if len(hit_times) == 0:
        print("ERROR: No valid hits loaded.")
        sys.exit(1)

    print(f"  {len(hit_times)} hits loaded")
    print(f"  Hit time range: {hit_times.min()*1E3:.4f} - {hit_times.max()*1E3:.4f} ms")

    print(f"\nWall Test Parameters:")
    print(f"  Vessel radius: {args.vessel_radius}")
    print(f"  Transmitter: ({tx_x}, {tx_y}, {tx_z})")
    print(f"  Receiver center: ({rx_x}, {rx_y}, {rx_z})")
    print(f"  3D distance (d): {d}")
    print(f"  Receiver radius (a): {a}")
    print(f"  Db: {args.db}")
    print(f"  Theoretical Pr(hit) as t->inf: a/d = {a/d:.4f}")

    all_pass = True

    # Test 1: Hit probability
    if args.total_paths:
        from scipy.stats import binomtest
        print(f"\n=== Test 1: Hit Probability (binomial) ===")
        p_theory = analytical_hit_probability(a, d, args.db, args.timestop)
        result = binomtest(len(hit_times), args.total_paths, p=p_theory, alternative='two-sided')
        empirical_rate = len(hit_times) / args.total_paths
        print(f"  Theoretical Pr(hit, t_stop={args.timestop}): {p_theory:.4f}")
        print(f"  Empirical hit rate: {empirical_rate:.4f} ({len(hit_times)}/{args.total_paths})")
        print(f"  Binomial test p-value: {result.pvalue:.6f}")
        binom_pass = result.pvalue > args.alpha
        print(f"  {'PASS' if binom_pass else 'FAIL'}")
        if not binom_pass:
            all_pass = False

    # Test 2: Hit-time distribution (KS test against conditional CDF)
    from scipy.stats import kstest
    print(f"\n=== Test 2: Hit-Time Distribution (KS, alpha={args.alpha}) ===")
    cdf_func = build_conditional_cdf(a, d, args.db, args.timestop)
    ks_stat, ks_pvalue = kstest(hit_times, cdf_func)
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {ks_pvalue:.6f}")
    ks_pass = ks_pvalue > args.alpha
    print(f"  {'PASS' if ks_pass else 'FAIL'}")
    if not ks_pass:
        all_pass = False

    print(f"\n{'PASS' if all_pass else 'FAIL'} - 3D wall reflection validation (KS p={ks_pvalue:.4f})")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        time_scale = 1E3
        tt = np.linspace(1E-8, args.timestop, 10000)
        pdf_analytical = h_diff_pdf(tt, a, d, args.db)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram vs analytical
        if len(hit_times) >= 1000:
            p_hit = analytical_hit_probability(a, d, args.db, args.timestop)
            counts, bin_edges = np.histogram(hit_times * time_scale, bins=120)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            trap_val = np.trapezoid(counts, bin_centers / time_scale)
            if trap_val > 0:
                pdf_sim = counts * (p_hit / trap_val)
                axes[0].plot(bin_centers, pdf_sim, 'k', linewidth=2, label='Simulation (walls)')
        axes[0].plot(tt * time_scale, pdf_analytical, 'r--', linewidth=2, label='Free-space analytical')
        axes[0].set_xlabel('time (msec)', fontsize=14)
        axes[0].set_ylabel('$H_{Diff}(t|d,t_0)$', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].set_title('Wall Reflection: PDF vs Free-Space')

        # CDF comparison
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
        plt.savefig('validation_3d_walls.png', dpi=150)
        print("\nPlot saved to validation_3d_walls.png")
        plt.show()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
