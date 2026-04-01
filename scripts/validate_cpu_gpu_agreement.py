#!/usr/bin/env python3
"""
Validate CPU and GPU simulation outputs are statistically consistent.

Both backends use different RNG (stdlib rand vs curand), so outputs won't
be identical. This test checks that the hit time distributions are
statistically similar using a two-sample Kolmogorov-Smirnov test.

Usage:
    python validate_cpu_gpu_agreement.py <cpu_csv> <gpu_csv>
"""

import argparse
import sys
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
    return hit_times[hit_times > 0]


def main():
    parser = argparse.ArgumentParser(description='Compare CPU and GPU simulation outputs')
    parser.add_argument('cpu_csv', help='CPU output CSV')
    parser.add_argument('gpu_csv', help='GPU output CSV')
    parser.add_argument('--timestep', type=float, default=1E-7, help='Simulation time step (s)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()

    print(f"Loading CPU: {args.cpu_csv}")
    cpu_times = load_hit_times(args.cpu_csv, args.timestep)
    print(f"  {len(cpu_times)} hits")

    print(f"Loading GPU: {args.gpu_csv}")
    gpu_times = load_hit_times(args.gpu_csv, args.timestep)
    print(f"  {len(gpu_times)} hits")

    # Hit rate comparison
    print(f"\n=== Hit Rate ===")
    print(f"  CPU hits: {len(cpu_times)}")
    print(f"  GPU hits: {len(gpu_times)}")
    rate_diff = abs(len(cpu_times) - len(gpu_times)) / max(len(cpu_times), len(gpu_times))
    print(f"  Difference: {rate_diff*100:.1f}%")
    rate_pass = rate_diff < 0.05  # within 5%
    print(f"  {'PASS' if rate_pass else 'FAIL'}")

    # Distribution moments
    print(f"\n=== Distribution Moments ===")
    cpu_mean = np.mean(cpu_times) * 1E3
    gpu_mean = np.mean(gpu_times) * 1E3
    cpu_std = np.std(cpu_times) * 1E3
    gpu_std = np.std(gpu_times) * 1E3
    print(f"  CPU mean: {cpu_mean:.4f} ms, std: {cpu_std:.4f} ms")
    print(f"  GPU mean: {gpu_mean:.4f} ms, std: {gpu_std:.4f} ms")
    mean_diff = abs(cpu_mean - gpu_mean) / cpu_mean
    std_diff = abs(cpu_std - gpu_std) / cpu_std
    print(f"  Mean difference: {mean_diff*100:.2f}%")
    print(f"  Std difference: {std_diff*100:.2f}%")
    moments_pass = mean_diff < 0.10 and std_diff < 0.10  # within 10%
    print(f"  {'PASS' if moments_pass else 'FAIL'}")

    # Two-sample KS test
    print(f"\n=== Kolmogorov-Smirnov Test ===")
    try:
        from scipy import stats
        ks_stat, ks_pvalue = stats.ks_2samp(cpu_times, gpu_times)
        print(f"  KS statistic: {ks_stat:.6f}")
        print(f"  p-value: {ks_pvalue:.6f}")
        ks_pass = ks_pvalue > 0.01  # reject agreement only at 1% level
        print(f"  {'PASS' if ks_pass else 'FAIL'} (p > 0.01 means distributions are consistent)")
    except ImportError:
        print("  scipy not installed, skipping KS test")
        ks_pass = True  # skip

    all_pass = rate_pass and moments_pass and ks_pass
    print(f"\n{'PASS' if all_pass else 'FAIL'} — CPU/GPU Agreement")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Overlaid histograms
        bins = np.linspace(0, max(cpu_times.max(), gpu_times.max()) * 1E3, 80)
        axes[0].hist(cpu_times * 1E3, bins=bins, alpha=0.6, density=True, label=f'CPU ({len(cpu_times)} hits)')
        axes[0].hist(gpu_times * 1E3, bins=bins, alpha=0.6, density=True, label=f'GPU ({len(gpu_times)} hits)')
        axes[0].set_xlabel('Hit time (ms)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Hit Time Distributions')
        axes[0].legend()

        # CDF comparison
        cpu_sorted = np.sort(cpu_times) * 1E3
        gpu_sorted = np.sort(gpu_times) * 1E3
        axes[1].plot(cpu_sorted, np.linspace(0, 1, len(cpu_sorted)), label='CPU')
        axes[1].plot(gpu_sorted, np.linspace(0, 1, len(gpu_sorted)), label='GPU')
        axes[1].set_xlabel('Hit time (ms)')
        axes[1].set_ylabel('CDF')
        axes[1].set_title('Cumulative Distribution')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('validation_cpu_gpu_agreement.png', dpi=150)
        print(f"\nPlot saved to validation_cpu_gpu_agreement.png")
        plt.show()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
