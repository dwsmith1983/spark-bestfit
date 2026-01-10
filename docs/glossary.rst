Glossary
========

Statistical and technical terms used throughout spark-bestfit documentation.

Goodness-of-Fit Metrics
-----------------------

.. glossary::
   :sorted:

   K-S statistic
   Kolmogorov-Smirnov statistic
      A measure of the maximum vertical distance between the empirical cumulative
      distribution function (ECDF) of the sample data and the theoretical CDF of
      the fitted distribution. Lower values indicate better fit.

      The K-S test is distribution-free and sensitive to both location and shape
      differences, making it a good general-purpose goodness-of-fit measure.

      **Range**: [0, 1] where 0 = perfect fit

      .. code-block:: python

         best = results.best(n=1)[0]  # Default: sorted by K-S
         print(f"K-S statistic: {best.ks_statistic}")

   A-D statistic
   Anderson-Darling statistic
      A goodness-of-fit statistic that measures discrepancies between the empirical
      and theoretical distributions, with more weight given to the tails compared
      to the :term:`K-S statistic`.

      Particularly useful when tail accuracy is important (e.g., risk analysis,
      extreme value modeling).

      **Range**: [0, infinity) where 0 = perfect fit

      .. code-block:: python

         best_ad = results.best(n=1, metric="ad_statistic")[0]
         print(f"A-D statistic: {best_ad.ad_statistic}")

   AIC
   Akaike Information Criterion
      A measure of relative model quality that balances goodness-of-fit against
      model complexity. Calculated as:

      ``AIC = 2k - 2ln(L)``

      where *k* is the number of parameters and *L* is the maximum likelihood.
      Lower AIC values indicate better models. Unlike :term:`K-S statistic`, AIC
      penalizes distributions with more parameters, helping prevent overfitting.

      **Use for**: Model selection when comparing distributions with different
      numbers of parameters. Recommended metric for discrete distributions.

   BIC
   Bayesian Information Criterion
      Similar to :term:`AIC` but with a stronger penalty for model complexity:

      ``BIC = k*ln(n) - 2ln(L)``

      where *n* is the sample size. BIC tends to select simpler models than AIC,
      especially with large sample sizes.

   SSE
   Sum of Squared Errors
      The sum of squared differences between the observed histogram frequencies
      and the fitted probability density function values at each bin. A simple,
      fast-to-compute fit quality metric.

      ``SSE = sum((observed - fitted)^2)``

      **Range**: [0, infinity) where 0 = perfect fit

Distribution Functions
----------------------

.. glossary::
   :sorted:

   PDF
   Probability Density Function
      For continuous distributions, the PDF gives the relative likelihood of the
      random variable taking a particular value. The area under the PDF curve
      over an interval gives the probability of falling within that interval.

      .. code-block:: python

         import numpy as np
         x = np.linspace(0, 10, 100)
         y = best.pdf(x)  # PDF values at points x

   CDF
   Cumulative Distribution Function
      The probability that a random variable takes a value less than or equal to
      a given value. CDF(x) = P(X <= x). The CDF is monotonically increasing from
      0 to 1.

      .. code-block:: python

         import numpy as np
         x = np.linspace(0, 10, 100)
         y = best.cdf(x)  # P(X <= x) for each x

   PPF
   Percent Point Function
   Quantile Function
   Inverse CDF
      The inverse of the :term:`CDF`. Given a probability *p* in [0, 1], the PPF
      returns the value *x* such that CDF(x) = p. Used for generating random
      samples and computing percentiles.

      - PPF(0.5) = median
      - PPF(0.25), PPF(0.75) = first and third quartiles
      - PPF(0.95) = 95th percentile

      .. code-block:: python

         median = best.ppf(0.5)
         percentile_95 = best.ppf(0.95)

      In spark-bestfit, the copula module uses ``fast_ppf`` for optimized inverse
      CDF computation during correlated sampling.

Multivariate Concepts
---------------------

.. glossary::
   :sorted:

   Copula
   Gaussian Copula
      A function that couples marginal distributions to form a joint multivariate
      distribution. The **Gaussian copula** uses a multivariate normal distribution
      to model dependence between variables while preserving their individual
      (marginal) distributions.

      In spark-bestfit, the Gaussian copula enables generating correlated samples
      across multiple columns while preserving each column's fitted marginal
      distribution.

      .. code-block:: python

         from spark_bestfit.copula import GaussianCopula

         # Fit distributions for each column
         results_a = fitter.fit(df, "col_a")
         results_b = fitter.fit(df, "col_b")

         # Create copula with fitted marginals
         copula = GaussianCopula([
             results_a.best(n=1)[0],
             results_b.best(n=1)[0]
         ])

         # Generate correlated samples
         samples = copula.sample(10000, df)

      See :doc:`features/copula` for details.

   Correlation Matrix
      A matrix showing the pairwise correlation coefficients between variables.
      Values range from -1 (perfect negative correlation) to +1 (perfect positive
      correlation), with 0 indicating no linear correlation. The Gaussian copula
      uses this matrix to model dependence structure.

Estimation Methods
------------------

.. glossary::
   :sorted:

   MLE
   Maximum Likelihood Estimation
      The default parameter estimation method used by scipy.stats distributions.
      MLE finds the parameter values that maximize the probability of observing
      the given data. Works well for most distributions with sufficient data.

   MSE
   Maximum Spacing Estimation
   Maximum Product of Spacings
      An alternative to :term:`MLE` that is more robust for heavy-tailed
      distributions. MSE maximizes the geometric mean of "spacings" (gaps between
      ordered observations transformed to uniform).

      Use MSE when MLE fails or produces unreasonable estimates for heavy-tailed
      data.

      .. code-block:: python

         from spark_bestfit import FitterConfigBuilder

         config = FitterConfigBuilder().with_mse_estimation().build()
         results = fitter.fit(df, "value", config=config)

      See :doc:`features/mse-estimation` for details.

   Bootstrap
   Bootstrap Confidence Intervals
      A resampling technique for estimating the uncertainty of fitted parameters.
      Repeatedly resamples the data with replacement, refits the distribution,
      and uses the distribution of fitted parameters to estimate confidence
      intervals.

      .. code-block:: python

         ci = best.confidence_intervals(df, "value", n_bootstrap=1000, alpha=0.05)
         # Returns 95% confidence intervals for each parameter

Distribution Properties
-----------------------

.. glossary::
   :sorted:

   Heavy-tailed Distribution
      A distribution whose tails decay slower than an exponential distribution.
      Heavy-tailed distributions have more probability mass in extreme values
      than normal-like distributions.

      Examples: Pareto, log-normal (sometimes), Cauchy, Student's t (low df)

      Heavy-tailed data often causes :term:`MLE` to fail. Use :term:`MSE` or
      specialized heavy-tail handling.

      See :doc:`features/heavy-tail` for detection and handling.

   Support
      The set of values where a distribution's :term:`PDF` is non-zero. For
      example:

      - Normal: (-infinity, +infinity)
      - Exponential: [0, +infinity)
      - Beta: [0, 1]
      - Uniform(a,b): [a, b]

      Use ``support_at_zero=True`` when fitting data that must be non-negative.

   Truncated Distribution
      A distribution restricted to a finite interval [a, b]. The original
      distribution's PDF is renormalized to integrate to 1 over the truncated
      range. Used for :doc:`bounded distribution fitting <features/bounded>`.

      .. code-block:: python

         # Fit distributions truncated to [0, 100]
         results = fitter.fit(df, "value", lower_bound=0, upper_bound=100)

Visualization Terms
-------------------

.. glossary::
   :sorted:

   Q-Q Plot
   Quantile-Quantile Plot
      A graphical method for comparing two distributions by plotting their
      quantiles against each other. If the distributions match, points fall
      along the diagonal line y=x.

      In spark-bestfit, Q-Q plots compare sample quantiles against theoretical
      quantiles from the fitted distribution. Deviations from the diagonal
      indicate poor fit.

      .. code-block:: python

         fitter.plot_qq(best, df, "value")

   P-P Plot
   Probability-Probability Plot
      Similar to a :term:`Q-Q plot` but plots cumulative probabilities instead
      of quantiles. The theoretical CDF values are plotted against empirical CDF
      values. Points should fall along the diagonal for a good fit.

      P-P plots are more sensitive to deviations in the middle of the distribution,
      while Q-Q plots are more sensitive to tail deviations.

      .. code-block:: python

         fitter.plot_pp(best, df, "value")

   Residual Plot
      A diagnostic plot showing the difference between observed and fitted values.
      Used to detect systematic patterns that indicate model misspecification.

      See :doc:`features/diagnostics-plots` for available diagnostic visualizations.

Spark/Backend Terms
-------------------

.. glossary::
   :sorted:

   Backend
      The execution engine used for parallel distribution fitting. spark-bestfit
      supports three backends:

      - **Local**: Default, uses Python multiprocessing
      - **Spark**: Apache Spark for production clusters
      - **Ray**: Ray for ML pipelines

      See :doc:`backends` for configuration.

   Broadcast Variable
      A Spark optimization that efficiently distributes read-only data to all
      executor nodes. spark-bestfit uses broadcast variables to share the data
      sample with all fitting tasks, avoiding repeated data transfers.

   Partition
      A subdivision of data for parallel processing. In Spark, each partition
      is processed by one task. More partitions = more parallelism but more
      scheduling overhead. spark-bestfit distributes distributions across
      partitions for parallel fitting.
