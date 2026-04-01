#!/usr/bin/env python3
"""
Validate that particle positions at a fixed time follow the expected
Gaussian distribution.

For constant-coefficient Brownian motion with drift, Euler-Maruyama
is exact: positions at time T are Gaussian with
  x ~ N(0, 2*D*T)
  y ~ N(0, 2*D*T)
  z ~ N(v*T, 2*D*T)

The key check is D-estimation: does Var(x)/(2*T) recover the input D?
This directly validates the stepping kernel coefficient sqrt(2*D*dt).

NOTE: This test is largely redundant with the first-passage time KS tests
(validate_1d_firsthit.py, validate_3d_diffusion.py). First-passage times
depend on the entire particle path, making them more sensitive to stepping
kernel errors than a single time-marginal. This test is kept as a fast
diagnostic — if it fails, the stepping kernel coefficient is wrong, which
is easier to diagnose than a first-passage test failure.

See docs/validation_methodology.md for full rationale.

Usage:
    python validate_position_distribution.py [--no-plot]
"""

import argparse
import subprocess
import sys
import os
import numpy as np


def run_simulation(binary, n_particles, n_steps, dt, velocity):
    """Run simulator in everything mode, return final positions.

    WARNING: This parses CPU output format only (particle-major: all steps
    for particle 0, then all steps for particle 1, etc.). GPU everything
    mode uses step-major format (all particles per line per timestep) and
    is NOT compatible with this parsing logic.

    NOTE: Diffusion coefficient (Db) is not configurable via CLI — it uses
    the hardcoded default from SimParams (1E-11). The expected variance
    calculation in the caller must match this value.
    """
    t_stop = n_steps * dt
    cmd = [
        binary,
        '-i', str(n_particles),
        '-e',
        '-t', f'{t_stop}',
        '-d', f'{dt}',
    ]
    if velocity == 0:
        cmd.append('-n')
    elif velocity != 1E-4:  # default
        cmd.extend(['--velocity', f'{velocity}'])

    binary_abs = os.path.abspath(binary)
    work_dir = os.path.dirname(binary_abs)
    cmd[0] = binary_abs
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        print(f"ERROR: Simulation failed: {result.stderr}")
        return None

    # Read output file
    output_path = os.path.join(work_dir, 'output_h.csv')
    data = np.genfromtxt(output_path, delimiter=',')
    if data.size == 0:
        return None

    # CPU everything mode: outer loop = particles (i), inner loop = steps (jj)
    # Layout: particle 0 all steps, particle 1 all steps, ...
    # Final position of particle i is at row (i+1)*n_steps - 1
    # NOTE: GPU everything mode has DIFFERENT layout — see run_simulation docstring
    expected_lines = n_particles * n_steps
    if len(data) < expected_lines:
        print(f"ERROR: Expected {expected_lines} lines, got {len(data)}")
        return None

    # Extract final position of each particle (last step of each particle's block)
    indices = np.arange(n_steps - 1, expected_lines, n_steps)
    final_positions = data[indices]

    # Columns: x, y, z (may have trailing NaN from comma)
    x = final_positions[:, 0]
    y = final_positions[:, 1]
    z = final_positions[:, 2]

    return x, y, z


def validate_axis(samples, expected_mean, expected_var, axis_name, T, alpha=0.001):
    """Validate one axis against N(expected_mean, expected_var).

    Tests three things:
      1. Mean matches expected (catches drift errors)
      2. Variance matches expected, i.e. D_estimated = Var/(2T) = D_input
         (catches coefficient errors in sqrt(2*D*dt))
      3. KS test against full Gaussian CDF (catches shape errors)

    Minimum n=20 for valid tests (Conover 1999).
    """
    from scipy.stats import kstest, norm

    n = len(samples)
    if n < 20:
        print(f"  (skipped — n={n} < 20)")
        return True

    sigma = np.sqrt(expected_var)
    frozen = norm(loc=expected_mean, scale=sigma)

    # KS test
    ks_stat, ks_pvalue = kstest(samples, frozen.cdf)

    # Mean comparison (catches drift magnitude/direction errors)
    sample_mean = np.mean(samples)
    mean_err = abs(sample_mean - expected_mean)
    mean_bound = 3 * sigma / np.sqrt(n)  # 3-sigma bound
    mean_pass = mean_err < mean_bound

    # Variance / D-estimation (catches coefficient errors in stepping kernel)
    sample_var = np.var(samples, ddof=1)
    D_estimated = sample_var / (2 * T) if T > 0 else float('inf')
    D_expected = expected_var / (2 * T) if T > 0 else 0
    var_ratio = sample_var / expected_var if expected_var > 0 else float('inf')
    var_bound_lo = 1 - 3 * np.sqrt(2.0 / (n - 1))
    var_bound_hi = 1 + 3 * np.sqrt(2.0 / (n - 1))
    var_pass = var_bound_lo < var_ratio < var_bound_hi

    print(f"\n  {axis_name}-axis (expected: mean={expected_mean:.2e}, var={expected_var:.2e}):")
    print(f"    Sample mean:  {sample_mean:.6e}  (err: {mean_err:.2e}, bound: {mean_bound:.2e})  {'PASS' if mean_pass else 'FAIL'}")
    print(f"    D estimated:  {D_estimated:.4e}  (expected: {D_expected:.4e}, ratio: {var_ratio:.4f})  {'PASS' if var_pass else 'FAIL'}")
    print(f"    KS stat: {ks_stat:.6f}, p-value: {ks_pvalue:.6f}  {'PASS' if ks_pvalue > alpha else 'FAIL'}")
    if n < 50:
        print(f"    (low-power: n={n} < 50)")

    return mean_pass and var_pass and (ks_pvalue > alpha)


def main():
    parser = argparse.ArgumentParser(description='Validate particle position distributions')
    parser.add_argument('--binary', default=None, help='Path to mc_sim_cpu binary')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--alpha', type=float, default=0.001, help='KS test significance level')
    args = parser.parse_args()

    # Find binary
    binary = args.binary
    if binary is None:
        for candidate in ['build/mc_sim_cpu', '../build/mc_sim_cpu']:
            if os.path.isfile(candidate):
                binary = candidate
                break
    if binary is None or not os.path.isfile(binary):
        print("ERROR: mc_sim_cpu binary not found. Build first or pass --binary.")
        sys.exit(1)

    all_pass = True

    # Test configuration using simulation defaults
    # Euler-Maruyama is exact for constant-coefficient SDEs, so this
    # directly tests the stepping kernel: does Var(x)/(2*T) = D?
    # (Allen & Tildesley; Frenkel & Smit — standard MD validation)
    #
    # Db default is 1E-11 in SimParams, configurable via --db flag.
    # This test uses the default — if changed, update db_expected here.
    db_expected = 1E-11

    tests = [
        {
            'name': 'Default params (D=1E-11, v=1E-4, T=1E-4)',
            'n_particles': 2000,
            'n_steps': 1000,  # T = 1000 * 1E-7 = 1E-4 s
            'dt': 1E-7,
            'velocity': 1E-4,
        },
    ]

    for test in tests:
        T = test['n_steps'] * test['dt']
        D = db_expected
        v = test['velocity']

        print(f"\n=== {test['name']} ===")
        print(f"  T={T}, D={D}, v={v}, n={test['n_particles']}")

        result = run_simulation(binary, test['n_particles'], test['n_steps'],
                                test['dt'], test['velocity'])
        if result is None:
            print("  FAIL — simulation error")
            all_pass = False
            continue

        x, y, z = result
        expected_var = 2 * D * T

        # x-axis: pure diffusion, N(0, 2*D*T)
        x_pass = validate_axis(x, 0.0, expected_var, 'x', T, args.alpha)

        # z-axis: diffusion + drift, N(v*T, 2*D*T)
        z_pass = validate_axis(z, v * T, expected_var, 'z', T, args.alpha)

        if not (x_pass and z_pass):
            all_pass = False

    print(f"\n{'PASS' if all_pass else 'FAIL'} - Position distribution validation")

    if not args.no_plot and all_pass:
        # Quick plot for the last test run
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, samples, label, mu, var in [
            (axes[0], x, 'x (diffusion only)', 0.0, expected_var),
            (axes[1], z, 'z (diffusion + drift)', v * T, expected_var),
        ]:
            sigma = np.sqrt(var)
            ax.hist(samples, bins=50, density=True, alpha=0.7, label='Simulation')
            t = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
            ax.plot(t, norm.pdf(t, mu, sigma), 'r-', lw=2, label=f'N({mu:.2e}, {var:.2e})')
            ax.set_xlabel(f'{label} position (m)')
            ax.set_ylabel('Density')
            ax.legend()

        plt.suptitle('Position Distribution at Fixed Time')
        plt.tight_layout()
        plt.savefig('validation_position_distribution.png', dpi=150)
        print("\nPlot saved to validation_position_distribution.png")
        plt.show()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
