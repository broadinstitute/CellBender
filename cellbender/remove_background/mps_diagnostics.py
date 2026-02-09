"""Diagnostic utilities for MPS posterior computation issues.

This module helps debug why MPS produces different noise estimates than CPU/CUDA.
The key finding is that per-cell background removal is ~0% on MPS vs ~21% on CPU,
even though training converges to similar loss values.

Usage:
    from cellbender.remove_background.mps_diagnostics import (
        diagnose_posterior_sampling,
        compare_guide_samples,
        analyze_lambda_distribution,
    )

    # After loading a checkpoint and creating posterior:
    results = diagnose_posterior_sampling(posterior, n_samples=20)
    print_full_diagnostic_report(results)
"""

import torch
import pyro
import pyro.distributions as dist
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('cellbender')


def diagnose_lambda_computation(posterior, n_samples: int = 5, n_cells: int = 10):
    """Diagnose lambda computation discrepancies between MPS and expected values.

    This function helps debug why MPS might produce different noise estimates
    than CPU/CUDA.

    Args:
        posterior: A Posterior object with a trained model
        n_samples: Number of samples to draw
        n_cells: Number of cells to analyze

    Returns:
        Dictionary with diagnostic information
    """
    import pyro

    model = posterior.vi_model
    device = model.device

    # Get some sample cell data
    count_matrix = posterior.dataset_obj.get_count_matrix()
    cell_inds = posterior.index_converter.analyzed_to_m[
        posterior.latents_map['p'] > 0.5
    ][:n_cells]

    if len(cell_inds) == 0:
        logger.warning("No cells found for diagnostics")
        return {}

    # Get the data for these cells
    data = torch.tensor(count_matrix[cell_inds].toarray(), dtype=torch.float32).to(device)

    # Collect diagnostic info
    diagnostics = {
        'device': str(device),
        'n_cells': len(cell_inds),
        'epsilon_samples': [],
        'd_empty_samples': [],
        'lambda_samples': [],
        'mu_samples': [],
    }

    # Get learned parameters
    chi_ambient = pyro.param('chi_ambient').detach()
    d_empty_loc = pyro.param('d_empty_loc').item()
    d_empty_scale = pyro.param('d_empty_scale').item()

    diagnostics['chi_ambient_sum'] = chi_ambient.sum().item()
    diagnostics['chi_ambient_max'] = chi_ambient.max().item()
    diagnostics['d_empty_loc'] = d_empty_loc
    diagnostics['d_empty_scale'] = d_empty_scale

    # Sample multiple times and collect lambda values
    for _ in range(n_samples):
        mu, lam, alpha = posterior.sample_mu_lambda_alpha(data, y_map=True)

        # Get epsilon from the guide trace (need to re-trace)
        guide_trace = pyro.poutine.trace(model.guide).get_trace(x=data)
        epsilon = guide_trace.nodes['epsilon']['value'].detach()
        d_empty = guide_trace.nodes['d_empty']['value'].detach()

        diagnostics['epsilon_samples'].append(epsilon.cpu().numpy())
        diagnostics['d_empty_samples'].append(d_empty.cpu().numpy())
        diagnostics['lambda_samples'].append(lam.cpu().numpy())
        diagnostics['mu_samples'].append(mu.cpu().numpy())

    # Compute statistics
    epsilon_arr = np.array(diagnostics['epsilon_samples'])
    d_empty_arr = np.array(diagnostics['d_empty_samples'])
    lambda_arr = np.array(diagnostics['lambda_samples'])
    mu_arr = np.array(diagnostics['mu_samples'])

    diagnostics['epsilon_mean'] = float(epsilon_arr.mean())
    diagnostics['epsilon_std'] = float(epsilon_arr.std())
    diagnostics['d_empty_mean'] = float(d_empty_arr.mean())
    diagnostics['d_empty_std'] = float(d_empty_arr.std())
    diagnostics['lambda_mean'] = float(lambda_arr.mean())
    diagnostics['lambda_std'] = float(lambda_arr.std())
    diagnostics['mu_mean'] = float(mu_arr.mean())
    diagnostics['mu_std'] = float(mu_arr.std())

    # Compute lambda/mu ratio (noise rate relative to signal)
    lambda_mu_ratio = lambda_arr.sum(axis=-1) / (mu_arr.sum(axis=-1) + 1e-10)
    diagnostics['lambda_mu_ratio_mean'] = float(lambda_mu_ratio.mean())
    diagnostics['lambda_mu_ratio_std'] = float(lambda_mu_ratio.std())

    # Clean up large arrays
    del diagnostics['epsilon_samples']
    del diagnostics['d_empty_samples']
    del diagnostics['lambda_samples']
    del diagnostics['mu_samples']

    return diagnostics


def verify_gamma_sampling(device: str = 'mps', n_samples: int = 10000):
    """Verify that gamma sampling on the specified device produces correct distributions.

    Args:
        device: Device to test ('cpu', 'cuda', 'mps')
        n_samples: Number of samples to draw

    Returns:
        Dictionary with test results
    """
    results = {'device': device, 'n_samples': n_samples, 'tests': {}}

    # Test various alpha values (these are typical for CellBender)
    # epsilon: concentration = enc['epsilon'] * 50 ≈ 25-125, rate = 50
    # phi: concentration = 1.0, rate = 5.0
    test_cases = [
        (0.5, 1.0, 'small_alpha'),
        (1.0, 1.0, 'unit_alpha'),
        (1.0, 5.0, 'phi_like'),  # phi prior
        (25.0, 50.0, 'epsilon_low'),  # low epsilon
        (50.0, 50.0, 'epsilon_mid'),  # mid epsilon
        (100.0, 50.0, 'epsilon_high'),  # high epsilon
    ]

    print(f"\nGamma Sampling Verification on {device}")
    print("-" * 50)

    for alpha_val, rate_val, name in test_cases:
        alpha = torch.tensor(alpha_val, device=device, dtype=torch.float32)
        rate = torch.tensor(rate_val, device=device, dtype=torch.float32)

        # Sample using the (potentially patched) Gamma distribution
        gamma_dist = dist.Gamma(alpha, rate)
        samples = gamma_dist.rsample((n_samples,))

        # Compute statistics
        mean = samples.mean().item()
        var = samples.var().item()

        # Expected values for Gamma(alpha, rate): mean = alpha/rate, var = alpha/rate^2
        expected_mean = alpha_val / rate_val
        expected_var = alpha_val / (rate_val ** 2)

        # Relative errors
        mean_error = abs(mean - expected_mean) / expected_mean
        var_error = abs(var - expected_var) / expected_var

        results['tests'][name] = {
            'alpha': alpha_val,
            'rate': rate_val,
            'mean': mean,
            'expected_mean': expected_mean,
            'mean_error': mean_error,
            'var': var,
            'expected_var': expected_var,
            'var_error': var_error,
        }

        status = "OK" if mean_error < 0.05 and var_error < 0.1 else "WARN"
        print(f"   {name}: Gamma({alpha_val}, {rate_val})")
        print(f"      mean={mean:.4f} (expected {expected_mean:.4f}, error {mean_error:.2%})")
        print(f"      var={var:.6f} (expected {expected_var:.6f}, error {var_error:.2%}) [{status}]")

    return results


def test_epsilon_sampling_for_cells(model, n_cells: int = 100, n_samples: int = 100):
    """Test epsilon sampling specifically for cell-like inputs.

    The guide samples epsilon with concentration gated by cell probability:
    epsilon_gated = prob * enc['epsilon'] + (1 - prob) * 1.0
    epsilon ~ Gamma(epsilon_gated * epsilon_prior, epsilon_prior)

    For cells (prob ≈ 1), concentration depends on the encoder output.

    Args:
        model: The trained model
        n_cells: Number of synthetic "cell" inputs to create
        n_samples: Number of samples per cell

    Returns:
        Dictionary with epsilon sampling statistics
    """
    device = model.device

    # Create synthetic cell data (high counts)
    # Use random high-count data that looks like cells
    torch.manual_seed(42)
    synthetic_cells = torch.rand(n_cells, model.n_genes, device=device) * 1000 + 100

    results = {
        'device': str(device),
        'n_cells': n_cells,
        'n_samples': n_samples,
        'epsilon_samples': [],
        'concentration_samples': [],
    }

    print(f"\nEpsilon Sampling Test for Cells on {device}")
    print("-" * 50)

    for _ in range(n_samples):
        # Get encoder output
        chi_ambient = pyro.param('chi_ambient').detach()
        enc = model.encoder(
            x=synthetic_cells,
            chi_ambient=chi_ambient,
            cell_prior_log=model.d_cell_loc_prior
        )

        # Compute epsilon_gated (what the guide uses)
        prob = enc['p_y'].sigmoid().detach()
        epsilon_gated = prob * enc['epsilon'] + (1 - prob) * 1.0

        # Sample epsilon from Gamma
        concentration = epsilon_gated * model.epsilon_prior
        rate = model.epsilon_prior
        epsilon = dist.Gamma(concentration, rate).rsample()

        results['epsilon_samples'].append(epsilon.cpu().numpy())
        results['concentration_samples'].append(concentration.cpu().numpy())

    # Compute statistics
    eps_arr = np.array(results['epsilon_samples'])
    conc_arr = np.array(results['concentration_samples'])

    results['epsilon_mean'] = float(eps_arr.mean())
    results['epsilon_std'] = float(eps_arr.std())
    results['epsilon_per_cell_mean'] = float(eps_arr.mean(axis=0).mean())
    results['epsilon_per_cell_std'] = float(eps_arr.mean(axis=0).std())

    results['concentration_mean'] = float(conc_arr.mean())
    results['concentration_std'] = float(conc_arr.std())

    # Expected epsilon = concentration / rate = epsilon_gated
    # So epsilon should be close to enc['epsilon'] for cells
    expected_epsilon = float(conc_arr.mean() / model.epsilon_prior.item())

    print(f"   Concentration: mean={results['concentration_mean']:.2f} ± {results['concentration_std']:.2f}")
    print(f"   Rate: {model.epsilon_prior.item():.1f}")
    print(f"   Expected epsilon: {expected_epsilon:.4f}")
    print(f"   Sampled epsilon: {results['epsilon_mean']:.4f} ± {results['epsilon_std']:.4f}")
    print(f"   Per-cell epsilon: {results['epsilon_per_cell_mean']:.4f} ± {results['epsilon_per_cell_std']:.4f}")

    # Clean up large arrays
    del results['epsilon_samples']
    del results['concentration_samples']

    return results


def run_full_diagnostic(posterior, n_cells: int = 20, n_samples: int = 50):
    """Run all diagnostics and print a comprehensive report.

    Args:
        posterior: A Posterior object with a trained model
        n_cells: Number of cells to analyze
        n_samples: Number of samples

    Returns:
        Dictionary with all diagnostic results
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MPS POSTERIOR DIAGNOSTIC")
    print("=" * 70)

    results = {}

    # 1. Verify gamma sampling
    print("\n[1/4] Verifying Gamma Sampling...")
    results['gamma'] = verify_gamma_sampling(str(posterior.vi_model.device), 10000)

    # 2. Test epsilon sampling for cells
    print("\n[2/4] Testing Epsilon Sampling for Cells...")
    results['epsilon'] = test_epsilon_sampling_for_cells(posterior.vi_model, 50, 50)

    # 3. Analyze lambda distribution
    print("\n[3/4] Analyzing Lambda Distribution...")
    results['lambda'] = analyze_lambda_distribution(posterior, n_cells, n_samples)

    # 4. Full posterior sampling diagnostic
    print("\n[4/4] Running Full Posterior Sampling Diagnostic...")
    results['posterior'] = diagnose_posterior_sampling(posterior, n_cells, n_samples)

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Check for issues
    issues = []

    # Check gamma sampling
    for name, test in results['gamma']['tests'].items():
        if test['mean_error'] > 0.1:
            issues.append(f"Gamma sampling error for {name}: mean error = {test['mean_error']:.1%}")

    # Check epsilon
    if results['epsilon']['epsilon_mean'] < 0.5 or results['epsilon']['epsilon_mean'] > 2.5:
        issues.append(f"Epsilon out of range: {results['epsilon']['epsilon_mean']:.4f}")

    # Check lambda
    if results['lambda']['lambda_total_per_cell_mean'] < 10:
        issues.append(f"Lambda very low: {results['lambda']['lambda_total_per_cell_mean']:.2f} counts/cell")

    if results['lambda']['lambda_mu_ratio_mean'] < 0.01:
        issues.append(f"Lambda/mu ratio very low: {results['lambda']['lambda_mu_ratio_mean']:.4f}")

    if issues:
        print("\nPOTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious issues found in sampling.")

    print("\n" + "=" * 70 + "\n")

    return results


def compare_guide_samples(model, data: torch.Tensor, n_samples: int = 50) -> Dict:
    """Sample from the guide multiple times and analyze the distributions.

    This is the key diagnostic: it shows what values are being sampled
    for epsilon, d_empty, phi during posterior computation.

    Args:
        model: The trained RemoveBackgroundPyroModel
        data: A batch of cell data (n_cells, n_genes)
        n_samples: Number of times to sample from the guide

    Returns:
        Dictionary with sample statistics for each latent variable
    """
    device = model.device
    results = {
        'device': str(device),
        'n_samples': n_samples,
        'n_cells': data.shape[0],
        'samples': {
            'epsilon': [],
            'd_empty': [],
            'phi': [],
            'd_cell': [],
            'y': [],
            'p_y': [],
        }
    }

    for _ in range(n_samples):
        # Trace the guide to get sampled values
        guide_trace = pyro.poutine.trace(model.guide).get_trace(x=data)

        # Extract sampled values
        if 'epsilon' in guide_trace.nodes:
            results['samples']['epsilon'].append(
                guide_trace.nodes['epsilon']['value'].detach().cpu().numpy()
            )
        if 'd_empty' in guide_trace.nodes:
            results['samples']['d_empty'].append(
                guide_trace.nodes['d_empty']['value'].detach().cpu().numpy()
            )
        if 'phi' in guide_trace.nodes:
            results['samples']['phi'].append(
                guide_trace.nodes['phi']['value'].detach().cpu().numpy()
            )
        if 'd_cell' in guide_trace.nodes:
            results['samples']['d_cell'].append(
                guide_trace.nodes['d_cell']['value'].detach().cpu().numpy()
            )
        if 'y' in guide_trace.nodes:
            results['samples']['y'].append(
                guide_trace.nodes['y']['value'].detach().cpu().numpy()
            )
        if 'p_passback' in guide_trace.nodes:
            results['samples']['p_y'].append(
                guide_trace.nodes['p_passback']['value'].detach().cpu().numpy()
            )

    # Compute statistics
    for key in results['samples']:
        if results['samples'][key]:
            arr = np.array(results['samples'][key])
            results[f'{key}_mean'] = float(arr.mean())
            results[f'{key}_std'] = float(arr.std())
            results[f'{key}_min'] = float(arr.min())
            results[f'{key}_max'] = float(arr.max())
            # Per-cell statistics (average across samples, then stats across cells)
            per_cell_mean = arr.mean(axis=0)
            results[f'{key}_per_cell_mean'] = float(per_cell_mean.mean())
            results[f'{key}_per_cell_std'] = float(per_cell_mean.std())

    return results


def analyze_lambda_distribution(posterior, n_cells: int = 20, n_samples: int = 50) -> Dict:
    """Analyze the distribution of lambda values during posterior computation.

    This directly examines what lambda values are being computed,
    which determines the noise estimates.

    Args:
        posterior: A Posterior object with a trained model
        n_cells: Number of cells to analyze
        n_samples: Number of samples per cell

    Returns:
        Dictionary with lambda statistics
    """
    model = posterior.vi_model
    device = model.device

    # Get some cell data
    count_matrix = posterior.dataset_obj.get_count_matrix()
    cell_mask = posterior.latents_map['p'] > 0.5
    cell_inds = np.where(cell_mask)[0][:n_cells]

    if len(cell_inds) == 0:
        return {'error': 'No cells found'}

    data = torch.tensor(
        count_matrix[cell_inds].toarray(),
        dtype=torch.float32
    ).to(device)

    # Get learned parameters
    chi_ambient = pyro.param('chi_ambient').detach()

    results = {
        'device': str(device),
        'n_cells': len(cell_inds),
        'n_samples': n_samples,
        'chi_ambient_sum': chi_ambient.sum().item(),
        'chi_ambient_mean': chi_ambient.mean().item(),
        'chi_ambient_max': chi_ambient.max().item(),
        'd_empty_loc': pyro.param('d_empty_loc').item(),
        'd_empty_scale': pyro.param('d_empty_scale').item(),
        'lambda_samples': [],
        'mu_samples': [],
        'alpha_samples': [],
        'epsilon_samples': [],
        'd_empty_samples': [],
    }

    # Sample multiple times
    for _ in range(n_samples):
        mu, lam, alpha = posterior.sample_mu_lambda_alpha(data, y_map=True)

        # Also get epsilon and d_empty from guide trace
        guide_trace = pyro.poutine.trace(model.guide).get_trace(x=data)
        epsilon = guide_trace.nodes['epsilon']['value'].detach()
        d_empty = guide_trace.nodes['d_empty']['value'].detach()

        results['lambda_samples'].append(lam.cpu().numpy())
        results['mu_samples'].append(mu.cpu().numpy())
        results['alpha_samples'].append(alpha.cpu().numpy() if alpha.dim() > 0 else np.array([alpha.item()]))
        results['epsilon_samples'].append(epsilon.cpu().numpy())
        results['d_empty_samples'].append(d_empty.cpu().numpy())

    # Compute statistics
    lambda_arr = np.array(results['lambda_samples'])
    mu_arr = np.array(results['mu_samples'])
    epsilon_arr = np.array(results['epsilon_samples'])
    d_empty_arr = np.array(results['d_empty_samples'])

    # Total lambda and mu per cell (sum over genes)
    lambda_total = lambda_arr.sum(axis=-1)  # [n_samples, n_cells]
    mu_total = mu_arr.sum(axis=-1)

    results['lambda_total_mean'] = float(lambda_total.mean())
    results['lambda_total_std'] = float(lambda_total.std())
    results['lambda_total_per_cell_mean'] = float(lambda_total.mean(axis=0).mean())
    results['lambda_total_per_cell_std'] = float(lambda_total.mean(axis=0).std())

    results['mu_total_mean'] = float(mu_total.mean())
    results['mu_total_std'] = float(mu_total.std())

    results['epsilon_mean'] = float(epsilon_arr.mean())
    results['epsilon_std'] = float(epsilon_arr.std())
    results['epsilon_per_cell_mean'] = float(epsilon_arr.mean(axis=0).mean())
    results['epsilon_per_cell_std'] = float(epsilon_arr.mean(axis=0).std())

    results['d_empty_mean'] = float(d_empty_arr.mean())
    results['d_empty_std'] = float(d_empty_arr.std())

    # Lambda/mu ratio (noise rate relative to signal)
    ratio = lambda_total / (mu_total + 1e-10)
    results['lambda_mu_ratio_mean'] = float(ratio.mean())
    results['lambda_mu_ratio_std'] = float(ratio.std())

    # Expected lambda from parameters
    # lambda = epsilon * d_empty * chi_ambient
    expected_d_empty = np.exp(results['d_empty_loc'] + 0.5 * results['d_empty_scale']**2)
    results['expected_d_empty'] = expected_d_empty
    results['expected_lambda_total'] = results['epsilon_mean'] * expected_d_empty * results['chi_ambient_sum']

    # Clean up large arrays
    del results['lambda_samples']
    del results['mu_samples']
    del results['alpha_samples']
    del results['epsilon_samples']
    del results['d_empty_samples']

    return results


def diagnose_posterior_sampling(posterior, n_cells: int = 20, n_samples: int = 50) -> Dict:
    """Comprehensive diagnostic of posterior sampling behavior.

    This is the main diagnostic function that checks all aspects of
    the posterior computation that could cause noise estimation issues.

    Args:
        posterior: A Posterior object
        n_cells: Number of cells to analyze
        n_samples: Number of samples

    Returns:
        Dictionary with comprehensive diagnostic results
    """
    model = posterior.vi_model
    device = model.device

    print(f"\n{'='*60}")
    print(f"POSTERIOR SAMPLING DIAGNOSTIC")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Get cell data
    count_matrix = posterior.dataset_obj.get_count_matrix()
    cell_mask = posterior.latents_map['p'] > 0.5
    cell_inds = np.where(cell_mask)[0][:n_cells]

    if len(cell_inds) == 0:
        print("ERROR: No cells found!")
        return {'error': 'No cells found'}

    data = torch.tensor(
        count_matrix[cell_inds].toarray(),
        dtype=torch.float32
    ).to(device)

    print(f"Analyzing {len(cell_inds)} cells with {n_samples} samples each\n")

    # 1. Check learned parameters
    print("1. LEARNED PARAMETERS")
    print("-" * 40)
    chi_ambient = pyro.param('chi_ambient').detach()
    d_empty_loc = pyro.param('d_empty_loc').item()
    d_empty_scale = pyro.param('d_empty_scale').item()
    phi_loc = pyro.param('phi_loc').item()
    phi_scale = pyro.param('phi_scale').item()

    print(f"   chi_ambient: sum={chi_ambient.sum().item():.4f}, max={chi_ambient.max().item():.6f}")
    print(f"   d_empty: loc={d_empty_loc:.4f}, scale={d_empty_scale:.4f}")
    print(f"   phi: loc={phi_loc:.4f}, scale={phi_scale:.4f}")
    print(f"   Expected d_empty value: {np.exp(d_empty_loc + 0.5*d_empty_scale**2):.2f}")
    print()

    # 2. Sample from guide and analyze
    print("2. GUIDE SAMPLING ANALYSIS")
    print("-" * 40)
    guide_results = compare_guide_samples(model, data, n_samples)

    for key in ['epsilon', 'd_empty', 'd_cell', 'phi']:
        if f'{key}_mean' in guide_results:
            print(f"   {key}: mean={guide_results[f'{key}_mean']:.4f} ± {guide_results[f'{key}_std']:.4f}")
            print(f"          range=[{guide_results[f'{key}_min']:.4f}, {guide_results[f'{key}_max']:.4f}]")
    print()

    # 3. Analyze lambda distribution
    print("3. LAMBDA (NOISE RATE) ANALYSIS")
    print("-" * 40)
    lambda_results = analyze_lambda_distribution(posterior, n_cells, n_samples)

    print(f"   Total lambda per cell: {lambda_results['lambda_total_per_cell_mean']:.4f} ± {lambda_results['lambda_total_per_cell_std']:.4f}")
    print(f"   Total mu per cell: {lambda_results['mu_total_mean']:.4f} ± {lambda_results['mu_total_std']:.4f}")
    print(f"   Lambda/mu ratio: {lambda_results['lambda_mu_ratio_mean']:.6f} ± {lambda_results['lambda_mu_ratio_std']:.6f}")
    print(f"   Expected lambda (from params): {lambda_results['expected_lambda_total']:.4f}")
    print()

    # 4. Check for systematic issues
    print("4. SYSTEMATIC CHECKS")
    print("-" * 40)

    # Check if lambda is too low
    if lambda_results['lambda_total_per_cell_mean'] < 1.0:
        print("   WARNING: Lambda is very low (< 1 count/cell)")
        print("            This will cause Poisson to strongly prefer noise=0")

    # Check lambda/mu ratio
    if lambda_results['lambda_mu_ratio_mean'] < 0.01:
        print("   WARNING: Lambda/mu ratio is very low (< 1%)")
        print("            Noise estimates will be near zero")

    # Check epsilon
    if guide_results.get('epsilon_mean', 1.0) < 0.5 or guide_results.get('epsilon_mean', 1.0) > 2.0:
        print(f"   WARNING: Epsilon mean ({guide_results.get('epsilon_mean', 'N/A'):.4f}) is outside expected range [0.5, 2.0]")

    # Check d_empty
    expected_d_empty = np.exp(d_empty_loc + 0.5*d_empty_scale**2)
    actual_d_empty = guide_results.get('d_empty_mean', expected_d_empty)
    if abs(actual_d_empty - expected_d_empty) / expected_d_empty > 0.2:
        print(f"   WARNING: Sampled d_empty ({actual_d_empty:.2f}) differs from expected ({expected_d_empty:.2f})")

    print()
    print("=" * 60)
    print()

    return {
        'guide_results': guide_results,
        'lambda_results': lambda_results,
        'params': {
            'chi_ambient_sum': chi_ambient.sum().item(),
            'd_empty_loc': d_empty_loc,
            'd_empty_scale': d_empty_scale,
            'phi_loc': phi_loc,
            'phi_scale': phi_scale,
        }
    }


def print_diagnostics_report(diagnostics: dict):
    """Print a formatted diagnostics report."""
    print("\n" + "="*60)
    print("MPS POSTERIOR DIAGNOSTICS REPORT")
    print("="*60)

    print(f"\nDevice: {diagnostics.get('device', 'unknown')}")
    print(f"Cells analyzed: {diagnostics.get('n_cells', 0)}")

    print("\n--- Learned Parameters ---")
    print(f"chi_ambient sum: {diagnostics.get('chi_ambient_sum', 'N/A'):.6f}")
    print(f"chi_ambient max: {diagnostics.get('chi_ambient_max', 'N/A'):.6f}")
    print(f"d_empty_loc: {diagnostics.get('d_empty_loc', 'N/A'):.4f}")
    print(f"d_empty_scale: {diagnostics.get('d_empty_scale', 'N/A'):.4f}")

    print("\n--- Sampled Values (mean ± std) ---")
    print(f"epsilon: {diagnostics.get('epsilon_mean', 'N/A'):.4f} ± {diagnostics.get('epsilon_std', 'N/A'):.4f}")
    print(f"d_empty: {diagnostics.get('d_empty_mean', 'N/A'):.4f} ± {diagnostics.get('d_empty_std', 'N/A'):.4f}")
    print(f"lambda (total): {diagnostics.get('lambda_mean', 'N/A'):.4f} ± {diagnostics.get('lambda_std', 'N/A'):.4f}")
    print(f"mu (total): {diagnostics.get('mu_mean', 'N/A'):.4f} ± {diagnostics.get('mu_std', 'N/A'):.4f}")

    print("\n--- Key Ratio ---")
    print(f"lambda/mu ratio: {diagnostics.get('lambda_mu_ratio_mean', 'N/A'):.6f} ± {diagnostics.get('lambda_mu_ratio_std', 'N/A'):.6f}")
    print("  (This ratio determines how much noise is estimated)")
    print("  (Low ratio = less noise estimated)")

    print("\n" + "="*60 + "\n")
