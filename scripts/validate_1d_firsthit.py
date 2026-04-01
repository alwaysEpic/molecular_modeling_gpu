#!/usr/bin/env python3
"""
Validate first-hitting time distribution against analytical solution.

Ported from TobyThesisTest_1_D_FirstHit.m (written by Dr. Uri Rogers).
Compares simulation CSV output against the 1-D inverse Gaussian passage
time distribution (thesis eq 4.3):

    p_passage(t) = b / sqrt(4*pi*Db*t^3) * exp(-(v*t - b)^2 / (4*Db*t))

Usage:
    python validate_1d_firsthit.py <csv_file> [--no-plot]
"""

import argparse
import sys
import numpy as np

def load_hit_times(filename, time_step=1E-7):
    """Load CSV and extract hit times from the timestep column."""
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Last non-NaN column contains the timestep index
    # CSV format: x, y, z, timestep,
    timesteps = data[:, 3]
    hit_times = timesteps * time_step
    return hit_times

def analytical_passage_time(t, b, Db, v):
    """1-D first passage time inverse Gaussian (thesis eq 4.3)."""
    return b / np.sqrt(4 * np.pi * Db * t**3) * np.exp(-(v * t - b)**2 / (4 * Db * t))

def validate(hit_times, b, Db, v, time_stop, num_bins=120):
    """Compare hit time histogram against analytical curve. Returns KS-like error metric."""
    time_scale = 1E3  # convert to msec for display

    # Histogram the simulation data
    counts, bin_edges = np.histogram(hit_times * time_scale, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    delta_t = (bin_centers[1] - bin_centers[0]) / time_scale
    # Normalize to PDF
    pdf_sim = counts / (np.sum(counts) * delta_t)

    # Analytical curve
    tt = np.linspace(1E-8, time_stop, 10000)
    pdf_analytical = analytical_passage_time(tt, b, Db, v)

    # Compute error: interpolate analytical at bin centers and compare
    from numpy import interp
    bin_centers_sec = bin_centers / time_scale
    pdf_analytical_at_bins = analytical_passage_time(bin_centers_sec, b, Db, v)

    # Only compare bins that have data
    mask = counts > 0
    if np.sum(mask) == 0:
        print("ERROR: No hits recorded in simulation output.")
        return None, None, None

    # Normalized RMS error (relative to peak)
    peak = np.max(pdf_analytical_at_bins[mask])
    rmse = np.sqrt(np.mean((pdf_sim[mask] - pdf_analytical_at_bins[mask])**2))
    nrmse = rmse / peak if peak > 0 else float('inf')

    return pdf_sim, bin_centers, nrmse

def main():
    parser = argparse.ArgumentParser(description='Validate 1-D first-hit times against analytical solution')
    parser.add_argument('csv_file', help='Path to simulation output CSV')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting, just print stats')
    parser.add_argument('--dist', type=float, default=3E-7, help='Distance to limit (m), default 3E-7')
    parser.add_argument('--db', type=float, default=1E-11, help='Diffusion coefficient (m^2/s)')
    parser.add_argument('--vel', type=float, default=1E-4, help='Drift velocity (m/s)')
    parser.add_argument('--timestep', type=float, default=1E-7, help='Simulation time step (s)')
    parser.add_argument('--timestop', type=float, default=1E-2, help='Simulation stop time (s)')
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    hit_times = load_hit_times(args.csv_file, args.timestep)
    print(f"  {len(hit_times)} hits loaded")
    print(f"  Hit time range: {hit_times.min()*1E3:.4f} - {hit_times.max()*1E3:.4f} ms")

    print(f"\nParameters:")
    print(f"  Distance (b): {args.dist}")
    print(f"  Db: {args.db}")
    print(f"  Velocity: {args.vel}")
    print(f"  Time step: {args.timestep}")

    pdf_sim, bin_centers, nrmse = validate(hit_times, args.dist, args.db, args.vel, args.timestop)

    if nrmse is None:
        sys.exit(1)

    print(f"\nNormalized RMSE (vs analytical): {nrmse:.4f}")
    if nrmse < 0.15:
        print("PASS - simulation matches analytical solution well")
    elif nrmse < 0.30:
        print("MARGINAL - rough agreement (may need more iterations)")
    else:
        print("FAIL - significant deviation from analytical solution")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        time_scale = 1E3
        tt = np.linspace(1E-8, args.timestop, 10000)
        pdf_analytical = analytical_passage_time(tt, args.dist, args.db, args.vel)

        fig, ax = plt.subplots(figsize=(8, 5))
        delta_t = (bin_centers[1] - bin_centers[0]) / time_scale
        ax.plot(bin_centers, pdf_sim, 'k', linewidth=2, label='Simulation')
        ax.plot(tt * time_scale, pdf_analytical, 'r--', linewidth=2, label='Analytical')
        ax.set_xlabel('time (msec)', fontsize=14)
        ax.set_ylabel('p_passage(t)', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_title(f'1-D First Hit Validation (NRMSE={nrmse:.4f})')
        plt.tight_layout()
        plt.savefig('validation_1d_firsthit.png', dpi=150)
        print("\nPlot saved to validation_1d_firsthit.png")
        plt.show()

if __name__ == '__main__':
    main()
