Real-World Use Cases
====================

This guide demonstrates spark-bestfit in production scenarios. Each use case includes
a complete Jupyter notebook with working code you can adapt to your needs.

.. contents:: Use Cases
   :local:
   :depth: 1

Monte Carlo Risk Simulation
---------------------------

**Business Context:** Financial risk managers need to estimate potential losses across
portfolios of correlated assets. Monte Carlo simulation generates thousands of scenarios
to calculate Value-at-Risk (VaR) and other risk metrics.

**spark-bestfit Features Used:**

- ``GaussianCopula`` for modeling asset correlations
- Multi-column fitting for portfolio assets
- Distributed sampling for scenario generation
- ``lazy_metrics=True`` for performance

**Notebook:** `examples/usecase_monte_carlo.ipynb <https://github.com/dwsmith1983/spark-bestfit/blob/main/examples/usecase_monte_carlo.ipynb>`_

.. code-block:: python

   from spark_bestfit import DistributionFitter, GaussianCopula

   # Fit distributions to historical returns
   fitter = DistributionFitter(spark)
   results = fitter.fit(returns_df, columns=["AAPL", "GOOGL", "MSFT"], lazy_metrics=True)

   # Model correlations from fit results
   copula = GaussianCopula.fit(results, returns_df)

   # Generate correlated scenarios
   scenarios = copula.sample(n_samples=10000, seed=42)

   # Calculate portfolio VaR
   portfolio_returns = scenarios.select(
       (F.col("AAPL") * 0.4 + F.col("GOOGL") * 0.35 + F.col("MSFT") * 0.25).alias("portfolio")
   )

ML Synthetic Data Generation
----------------------------

**Business Context:** Machine learning teams need synthetic data for model training,
testing, and privacy-preserving data sharing. Fitting distributions to real data
enables generating statistically similar synthetic datasets.

**spark-bestfit Features Used:**

- Multi-column fitting for feature columns
- ``DiscreteDistributionFitter`` for categorical/count features
- Serialization for saving/loading fitted models
- Distributed sampling at scale

**Notebook:** `examples/usecase_synthetic_data.ipynb <https://github.com/dwsmith1983/spark-bestfit/blob/main/examples/usecase_synthetic_data.ipynb>`_

.. code-block:: python

   from spark_bestfit import DistributionFitter, DiscreteDistributionFitter

   # Fit continuous features
   cont_fitter = DistributionFitter(spark)
   cont_results = cont_fitter.fit(df, columns=["age", "income", "score"])

   # Fit discrete features
   disc_fitter = DiscreteDistributionFitter(spark)
   disc_results = disc_fitter.fit(df, columns=["num_purchases", "category_id"])

   # Save for reproducibility
   cont_results.save("models/continuous_fits.json")

   # Generate synthetic data
   synthetic_df = sample_from_results(cont_results, n=100000)

A/B Test Analysis
-----------------

**Business Context:** Product teams run experiments to measure the impact of changes.
Distribution fitting helps model conversion rates, revenue per user, and other metrics
with proper uncertainty quantification.

**spark-bestfit Features Used:**

- Bounded fitting for proportions (0-1 range)
- Bootstrap confidence intervals
- ``lazy_metrics=True`` for quick model selection

**Notebook:** `examples/usecase_ab_testing.ipynb <https://github.com/dwsmith1983/spark-bestfit/blob/main/examples/usecase_ab_testing.ipynb>`_

.. code-block:: python

   from spark_bestfit import DistributionFitter

   fitter = DistributionFitter(spark)

   # Fit bounded distributions to conversion rates
   # Beta naturally fits [0, 1] bounded data
   results = fitter.fit(
       experiment_df,
       column="conversion_rate",
       bounded=True,
       lower_bound=0.0,
       upper_bound=1.0,
       lazy_metrics=True
   )

   # Get best fit (typically Beta for proportions)
   best = results.best(n=1, metric='aic')[0]
   samples = best.sample(size=10000)  # For bootstrap CI

Insurance Claims Modeling
-------------------------

**Business Context:** Actuaries model claim frequency and severity to set premiums
and reserves. Heavy-tailed distributions (Pareto, lognormal) are essential for
capturing extreme loss events.

**spark-bestfit Features Used:**

- Heavy-tail distributions (Pareto, lognormal, Weibull)
- ``DiscreteDistributionFitter`` for claim counts
- Bounded fitting for capped policies
- Q-Q plots for tail behavior validation

**Notebook:** `examples/usecase_insurance.ipynb <https://github.com/dwsmith1983/spark-bestfit/blob/main/examples/usecase_insurance.ipynb>`_

.. code-block:: python

   from spark_bestfit import DistributionFitter, DiscreteDistributionFitter

   # Fit claim severity (includes heavy-tailed: pareto, lognorm, burr, etc.)
   severity_fitter = DistributionFitter(spark)
   severity_results = severity_fitter.fit(
       claims_df,
       column="claim_amount",
       lazy_metrics=True
   )

   # Fit claim frequency (discrete: poisson, nbinom, etc.)
   freq_fitter = DiscreteDistributionFitter(spark)
   freq_results = freq_fitter.fit(claims_df, column="num_claims")

   # Get best fits and visualize
   best_severity = severity_results.best(n=1, metric='aic')[0]
   severity_fitter.plot(best_severity, claims_df, "claim_amount")

Risk Model Validation
---------------------

**Business Context:** Financial regulators require statistical validation of risk models.
Unlike model *selection* (AIC), model *validation* asks: "Does the data actually come
from this distribution?" The Kolmogorov-Smirnov (KS) test provides formal hypothesis testing.

**spark-bestfit Features Used:**

- ``lazy_metrics=False`` to compute KS and Anderson-Darling statistics
- ``metric='ks_statistic'`` for goodness-of-fit based selection
- ``metric='ad_statistic'`` for tail-sensitive validation
- ``fit.pvalue`` for hypothesis test interpretation

**Notebook:** `examples/usecase_model_validation.ipynb <https://github.com/dwsmith1983/spark-bestfit/blob/main/examples/usecase_model_validation.ipynb>`_

.. code-block:: python

   from spark_bestfit import DistributionFitter

   fitter = DistributionFitter(spark)

   # Fit with full metrics (not lazy) for validation
   results = fitter.fit(
       returns_df,
       column="daily_return",
       lazy_metrics=False  # Compute KS and AD statistics
   )

   # Select by goodness-of-fit (not prediction accuracy)
   best = results.best(n=1, metric='ks_statistic')[0]

   # Interpret hypothesis test
   if best.pvalue > 0.05:
       print("Model PASSES validation (cannot reject H0)")
   else:
       print("Model FAILS validation (reject H0)")

Discrete Event Simulation
-------------------------

**Business Context:** Operations teams need to answer "what-if" questions about
staffing, capacity, and process changes. Rather than experimenting with real
operations (expensive and risky), you can fit distributions to historical data
and simulate scenarios.

**spark-bestfit Features Used:**

- ``DistributionFitter`` for inter-arrival and service times
- ``DiscreteDistributionFitter`` for hourly/daily volumes
- ``lazy_metrics=False`` to validate distributional assumptions
- ``get_scipy_dist()`` to sample from fitted distributions in simulations

**Notebook:** `examples/usecase_simulation.ipynb <https://github.com/dwsmith1983/spark-bestfit/blob/main/examples/usecase_simulation.ipynb>`_

.. code-block:: python

   from spark_bestfit import DistributionFitter

   fitter = DistributionFitter(spark)

   # Fit distributions to operational data
   arrival_results = fitter.fit(df, column='inter_arrival_seconds', lazy_metrics=False)
   service_results = fitter.fit(df, column='service_time_seconds', lazy_metrics=False)

   # Get best fits for simulation
   arrival_dist = arrival_results.best(n=1, metric='aic')[0].get_scipy_dist()
   service_dist = service_results.best(n=1, metric='aic')[0].get_scipy_dist()

   # Simulate queue with fitted distributions
   inter_arrivals = arrival_dist.rvs(size=1000)
   service_times = service_dist.rvs(size=1000)

   # Run what-if scenarios (add agents, change volume, etc.)

Which Use Case Fits Your Needs?
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Use Case
     - Key Features
     - Best For
   * - Monte Carlo
     - Copula, sampling
     - Risk management, finance, simulations
   * - Synthetic Data
     - Multi-column, serialization
     - ML training, privacy, testing
   * - A/B Testing
     - Bounded, bootstrap CI
     - Product experiments, marketing
   * - Insurance
     - Heavy-tail, discrete
     - Actuarial, loss modeling
   * - Model Validation
     - KS/AD tests, p-values
     - Regulatory compliance, backtesting
   * - Discrete Event Simulation
     - get_scipy_dist(), sampling
     - Operations, staffing, capacity planning

See Also
--------

- :doc:`quickstart` - Basic usage and installation
- :doc:`copula` - Detailed copula documentation
- :doc:`sampling` - Distributed sampling guide
- :doc:`bounded` - Bounded distribution fitting
