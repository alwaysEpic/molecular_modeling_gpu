#!/usr/bin/env python3
"""
Validate 3D diffusion-only first-hit time against analytical solution.

Ported from TobyThesisTest.m (written by Dr. Uri Rogers).
Compares simulation CSV output against the 3D spherical receiver
hit probability (thesis eq 4.1, Schulten eq 3.108):

    H_Diff(t|d,t0) = (a/d) * 1/sqrt(4*pi*D*(t-t0)) * (d-a)/(t-t0) * exp(-(d-a)^2 / (4*D*(t-t0)))

Usage:
    python validate_3d_diffusion.py <csv_file> [--no-plot]
"""

import argparse
import sys
import numpy as np


def load_hit_times(filename, time_step=1E-7):
    """Load CSV and extract hit times from the timestep column."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    valid_cols = np.sum(~np.isnan(data[0, :]))
    if valid_cols >= 5:
        timesteps = data[:, 4]  # CPU output
    else:
        timesteps = data[:, 3]  # GPU output
    hit_times = timesteps * time_step
    return hit_times


def analytical_hit_pdf(t, a, d, Db):
    """3D diffusion-only first-hitting time pdf (thesis eq 4.1)."""
    return (a / d) * (1.0 / np.sqrt(4 * np.pi * Db * t)) * ((d - a) / t) * np.exp(-(d - a)**2 / (4 * Db * t))


def analytical_hit_probability(a, d, Db, t_stop):
    """Asymptotic hit probability (thesis eq 4.2, Schulten eq 3.116)."""
    from scipy.special import erfc
    return (a / d) * erfc((d - a) / np.sqrt(4 * Db * t_stop))


def validate(hit_times, a, d, Db, t_stop, num_bins=120):
    """Compare hit time histogram against analytical curve."""
    time_scale = 1E3  # msec

    prob_hit = analytical_hit_probability(a, d, Db, t_stop)

    counts, bin_edges = np.histogram(hit_times * time_scale, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    delta_t = (bin_centers[1] - bin_centers[0]) / time_scale

    # Scale histogram by hit probability (same as thesis MATLAB script)
    pdf_sim = counts * (prob_hit / np.trapezoid(counts, bin_centers / time_scale))

    # Analytical curve
    tt = np.linspace(1E-8, t_stop, 10000)
    pdf_analytical = analytical_hit_pdf(tt, a, d, Db)

    # Compare at bin centers
    bin_centers_sec = bin_centers / time_scale
    pdf_analytical_at_bins = analytical_hit_pdf(bin_centers_sec, a, d, Db)

    mask = counts > 0
    if np.sum(mask) == 0:
        print("ERROR: No hits recorded.")
        return None, None, None, prob_hit

    peak = np.max(pdf_analytical_at_bins[mask])
    rmse = np.sqrt(np.mean((pdf_sim[mask] - pdf_analytical_at_bins[mask])**2))
    nrmse = rmse / peak if peak > 0 else float('inf')

    return pdf_sim, bin_centers, nrmse, prob_hit


def main():
    parser = argparse.ArgumentParser(description='Validate 3D diffusion-only first-hit times')
    parser.add_argument('csv_file', help='Path to simulation output CSV')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--receiver-radius', type=float, default=10E-9, help='Receiver radius (m), default 10nm')
    parser.add_argument('--distance', type=float, default=50E-9, help='Transmitter-receiver distance (m), default 50nm')
    parser.add_argument('--db', type=float, default=1E-11, help='Diffusion coefficient (m^2/s)')
    parser.add_argument('--timestep', type=float, default=1E-7, help='Simulation time step (s)')
    parser.add_argument('--timestop', type=float, default=1E-3, help='Simulation stop time (s)')
    parser.add_argument('--total-paths', type=int, default=None, help='Total paths simulated (for hit rate check)')
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    hit_times = load_hit_times(args.csv_file, args.timestep)
    print(f"  {len(hit_times)} hits loaded")
    print(f"  Hit time range: {hit_times.min()*1E3:.4f} - {hit_times.max()*1E3:.4f} ms")

    a = args.receiver_radius
    d = args.distance
    print(f"\nParameters:")
    print(f"  Receiver radius (a): {a}")
    print(f"  Distance (d): {d}")
    print(f"  Db: {args.db}")
    print(f"  Theoretical Pr(hit) = a/d = {a/d:.4f}")

    pdf_sim, bin_centers, nrmse, prob_hit = validate(
        hit_times, a, d, args.db, args.timestop)

    if nrmse is None:
        sys.exit(1)

    print(f"  Computed Pr(hit) with erfc: {prob_hit:.4f}")
    if args.total_paths:
        empirical_rate = len(hit_times) / args.total_paths
        print(f"  Empirical hit rate: {empirical_rate:.4f} ({len(hit_times)}/{args.total_paths})")

    print(f"\nNormalized RMSE (vs analytical): {nrmse:.4f}")
    if nrmse < 0.20:
        print("PASS - simulation matches analytical solution well")
    elif nrmse < 0.40:
        print("MARGINAL - rough agreement (may need more iterations)")
    else:
        print("FAIL - significant deviation from analytical solution")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        time_scale = 1E3
        tt = np.linspace(1E-8, args.timestop, 10000)
        pdf_analytical = analytical_hit_pdf(tt, a, d, args.db)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bin_centers, pdf_sim, 'k', linewidth=2, label='Simulation')
        ax.plot(tt * time_scale, pdf_analytical, 'r--', linewidth=2, label='Analytical')
        ax.set_xlabel('time (msec)', fontsize=14)
        ax.set_ylabel('$H_{Diff}(t|d,t_0)$', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_title(f'3D Diffusion-Only Validation (NRMSE={nrmse:.4f})')
        plt.tight_layout()
        plt.savefig('validation_3d_diffusion.png', dpi=150)
        print("\nPlot saved to validation_3d_diffusion.png")
        plt.show()


if __name__ == '__main__':
    main()
