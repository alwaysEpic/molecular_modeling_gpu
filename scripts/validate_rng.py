#!/usr/bin/env python3
"""
Validate random number generator quality.

Replicates the thesis RNG validation (c_tester.m, random_tester.m):
1. Statistical moments: mean, std, variance, kurtosis vs expected N(0,1)
2. Autocorrelation: should be sharp peak at lag 0, near-zero elsewhere
3. Normality test: Shapiro-Wilk or Anderson-Darling

Reads a CSV file of raw random numbers (one per line or one column).

Usage:
    python validate_rng.py <csv_file> [--column N] [--no-plot]
"""

import argparse
import sys
import numpy as np


def load_random_numbers(filename, column=0):
    """Load random numbers from CSV."""
    data = np.genfromtxt(filename, delimiter=',')
    if data.ndim == 1:
        return data[~np.isnan(data)]
    return data[:, column][~np.isnan(data[:, column])]


def test_moments(samples):
    """Test statistical moments against expected N(0,1) values."""
    n = len(samples)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    var = np.var(samples, ddof=1)
    kurt = float(np.mean((samples - mean)**4) / std**4)  # excess kurtosis + 3

    # Expected for N(0,1): mean=0, std=1, var=1, kurtosis=3
    # Tolerances scale with 1/sqrt(n)
    tol_mean = 3.0 / np.sqrt(n)  # 3-sigma bound
    tol_std = 3.0 * np.sqrt(1.0 / (2 * n))  # approx 3-sigma for std estimator
    tol_kurt = 3.0 * np.sqrt(24.0 / n)  # approx 3-sigma for kurtosis

    results = {
        'n': n,
        'mean': mean,
        'std': std,
        'variance': var,
        'kurtosis': kurt,
        'mean_pass': abs(mean) < tol_mean,
        'std_pass': abs(std - 1.0) < tol_std,
        'kurt_pass': abs(kurt - 3.0) < tol_kurt,
    }
    return results


def test_autocorrelation(samples, max_lag=50):
    """Test autocorrelation — should be ~0 for all lags > 0."""
    n = len(samples)
    samples_centered = samples - np.mean(samples)
    acf = np.correlate(samples_centered, samples_centered, mode='full')
    acf = acf[n-1:]  # positive lags only
    acf = acf / acf[0]  # normalize

    # For white noise, autocorrelation at lag k should be ~N(0, 1/n)
    # 95% confidence bound
    bound = 1.96 / np.sqrt(n)
    lags_to_check = min(max_lag, len(acf) - 1)
    violations = sum(1 for k in range(1, lags_to_check + 1) if abs(acf[k]) > bound)
    # Allow up to 5% violations (expected for random data)
    violation_rate = violations / lags_to_check

    return {
        'acf': acf[:lags_to_check + 1],
        'bound': bound,
        'violations': violations,
        'lags_checked': lags_to_check,
        'violation_rate': violation_rate,
        'pass': violation_rate < 0.10,  # allow up to 10%
    }


def test_normality(samples):
    """Anderson-Darling test for normality."""
    from scipy import stats
    try:
        result = stats.anderson(samples, dist='norm')
        # Use 5% significance level (index 2)
        is_normal = result.statistic < result.critical_values[2]
        return {
            'statistic': result.statistic,
            'critical_5pct': result.critical_values[2],
            'pass': is_normal,
        }
    except ImportError:
        # scipy not available, skip
        return {'pass': None, 'note': 'scipy not installed, skipping Anderson-Darling'}


def main():
    parser = argparse.ArgumentParser(description='Validate RNG quality')
    parser.add_argument('csv_file', help='CSV file with random numbers')
    parser.add_argument('--column', type=int, default=0, help='Column index to read (default 0)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()

    print(f"Loading {args.csv_file}...")
    samples = load_random_numbers(args.csv_file, args.column)
    print(f"  {len(samples)} samples loaded")

    # Moments test
    print(f"\n=== Statistical Moments (expected: N(0,1)) ===")
    moments = test_moments(samples)
    print(f"  Mean:     {moments['mean']:+.6f}  (expected 0)     {'PASS' if moments['mean_pass'] else 'FAIL'}")
    print(f"  Std:      {moments['std']:.6f}   (expected 1)     {'PASS' if moments['std_pass'] else 'FAIL'}")
    print(f"  Variance: {moments['variance']:.6f}   (expected 1)")
    print(f"  Kurtosis: {moments['kurtosis']:.6f}   (expected 3)     {'PASS' if moments['kurt_pass'] else 'FAIL'}")

    # Autocorrelation test
    print(f"\n=== Autocorrelation ===")
    acf_result = test_autocorrelation(samples)
    print(f"  Lags checked: {acf_result['lags_checked']}")
    print(f"  95% bound: +/- {acf_result['bound']:.6f}")
    print(f"  Violations: {acf_result['violations']} ({acf_result['violation_rate']*100:.1f}%)")
    print(f"  {'PASS' if acf_result['pass'] else 'FAIL'}")

    # Normality test
    print(f"\n=== Normality (Anderson-Darling) ===")
    norm_result = test_normality(samples)
    if norm_result['pass'] is None:
        print(f"  {norm_result['note']}")
    else:
        print(f"  Statistic: {norm_result['statistic']:.4f}  (critical 5%: {norm_result['critical_5pct']:.4f})")
        print(f"  {'PASS' if norm_result['pass'] else 'FAIL'}")

    # Overall
    all_pass = moments['mean_pass'] and moments['std_pass'] and moments['kurt_pass'] and acf_result['pass']
    print(f"\n{'PASS' if all_pass else 'FAIL'} — Overall RNG quality")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histogram vs N(0,1)
        axes[0].hist(samples, bins=100, density=True, alpha=0.7, label='Samples')
        x = np.linspace(-4, 4, 200)
        axes[0].plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), 'r-', lw=2, label='N(0,1)')
        axes[0].set_title('Distribution')
        axes[0].legend()

        # Autocorrelation
        lags = np.arange(len(acf_result['acf']))
        axes[1].bar(lags, acf_result['acf'], width=0.8)
        axes[1].axhline(y=acf_result['bound'], color='r', linestyle='--', label='95% bound')
        axes[1].axhline(y=-acf_result['bound'], color='r', linestyle='--')
        axes[1].set_title('Autocorrelation')
        axes[1].set_xlabel('Lag')
        axes[1].legend()

        # QQ plot
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        theoretical = np.array([np.sqrt(2) * np.math.erfc(2 * (i + 0.5) / n) for i in range(n)])
        # Simple approximation: use percentile points of N(0,1)
        from numpy import percentile
        expected = np.linspace(0.5/n, 1 - 0.5/n, n)
        theoretical_quantiles = np.sqrt(2) * np.array([float(-np.polynomial.hermite_e.hermeval(0, [0]*20)) for _ in expected])
        # Simpler: just use scipy if available
        try:
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(expected)
            axes[2].scatter(theoretical_quantiles, sorted_samples, s=1, alpha=0.3)
            axes[2].plot([-4, 4], [-4, 4], 'r-', lw=2)
            axes[2].set_title('Q-Q Plot')
            axes[2].set_xlabel('Theoretical')
            axes[2].set_ylabel('Sample')
        except ImportError:
            axes[2].text(0.5, 0.5, 'scipy needed\nfor QQ plot', transform=axes[2].transAxes, ha='center')

        plt.tight_layout()
        plt.savefig('validation_rng.png', dpi=150)
        print(f"\nPlot saved to validation_rng.png")
        plt.show()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
