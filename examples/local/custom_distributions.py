#!/usr/bin/env python3
"""Custom distribution fitting example.

This example demonstrates how to register and fit custom scipy rv_continuous
distributions alongside the built-in scipy.stats distributions.

Usage:
    python custom_distributions.py
"""

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous

from spark_bestfit import DistributionFitter, LocalBackend


class PowerDistribution(rv_continuous):
    """Power distribution: f(x) = alpha * x^(alpha-1) for x in [0, 1].

    This is a simple one-parameter distribution useful for modeling
    data concentrated near 0 or 1 depending on alpha.

    - alpha < 1: density peaks at 0
    - alpha = 1: uniform distribution
    - alpha > 1: density peaks at 1
    """

    def _pdf(self, x, alpha):
        return alpha * np.power(x, alpha - 1)

    def _cdf(self, x, alpha):
        return np.power(x, alpha)


class ShiftedExponential(rv_continuous):
    """Exponential distribution shifted by a location parameter.

    f(x) = rate * exp(-rate * (x - loc)) for x >= loc

    Useful for modeling waiting times with a minimum threshold.
    """

    def _pdf(self, x, rate):
        return rate * np.exp(-rate * x)

    def _cdf(self, x, rate):
        return 1 - np.exp(-rate * x)


def main():
    print("=" * 60)
    print("Custom Distribution Fitting Example")
    print("=" * 60)

    # Generate data from a power distribution (alpha=2)
    # Using inverse CDF: x = u^(1/alpha) where u ~ Uniform(0,1)
    np.random.seed(42)
    alpha_true = 2.0
    u = np.random.uniform(0, 1, 2000)
    data = u ** (1 / alpha_true)

    df = pd.DataFrame({"value": data})
    print(f"\nGenerated {len(df)} samples from Power(alpha={alpha_true})")
    print(f"  Data range: [{df['value'].min():.3f}, {df['value'].max():.3f}]")
    print(f"  Mean: {df['value'].mean():.3f} (theoretical: {alpha_true/(alpha_true+1):.3f})")

    # Create backend and fitter
    backend = LocalBackend(max_workers=4)
    fitter = DistributionFitter(backend=backend)

    # Register custom distributions
    # Note: support bounds are set in constructor (a=lower, b=upper)
    power_dist = PowerDistribution(a=0, b=1, name="power")
    shifted_exp = ShiftedExponential(a=0, b=np.inf, name="shifted_exp")

    fitter.register_distribution("power", power_dist)
    fitter.register_distribution("shifted_exp", shifted_exp)

    print(f"\nRegistered custom distributions: {list(fitter.get_custom_distributions().keys())}")

    # Fit all distributions (including our custom ones)
    print("\nFitting distributions...")
    results = fitter.fit(df, column="value", max_distributions=20)

    # Show top results
    print("\nTop 10 distributions (by SSE):")
    print("-" * 60)
    for i, result in enumerate(results.best(n=10, metric="sse"), 1):
        marker = " <-- CUSTOM" if result.distribution in ["power", "shifted_exp"] else ""
        print(f"{i:2}. {result.distribution:20} SSE={result.sse:.6f}{marker}")

    # Check if our power distribution won (it should!)
    best = results.best(n=1, metric="sse")[0]
    print(f"\nBest fit: {best.distribution}")

    if best.distribution == "power":
        print("   Our custom Power distribution won!")
        alpha_fitted = best.parameters[0]
        print(f"   Fitted alpha: {alpha_fitted:.3f} (true: {alpha_true:.1f})")
    else:
        # Find our power distribution result
        power_results = [r for r in results.best(n=100) if r.distribution == "power"]
        if power_results:
            power_result = power_results[0]
            print(f"\n   Power distribution result:")
            print(f"   Fitted alpha: {power_result.parameters[0]:.3f}")
            print(f"   SSE: {power_result.sse:.6f}")

    # Generate samples from the best fit
    print(f"\nSampling from best fit ({best.distribution}):")
    samples = best.sample(500)
    print(f"   Sample mean: {samples.mean():.3f}")
    print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Demonstrate method chaining
    print("\n" + "=" * 60)
    print("Method Chaining Example")
    print("=" * 60)

    chained_fitter = (
        DistributionFitter(backend=LocalBackend())
        .register_distribution("power", PowerDistribution(a=0, b=1))
        .register_distribution("shifted_exp", ShiftedExponential(a=0, b=np.inf))
    )
    print(f"Registered via chaining: {list(chained_fitter.get_custom_distributions().keys())}")


if __name__ == "__main__":
    main()
