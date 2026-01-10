Glossary
========

Statistical and technical terms used throughout the spark-bestfit documentation.

.. glossary::
   :sorted:

   AIC
      **Akaike Information Criterion**. A metric that balances goodness-of-fit against
      model complexity. Lower AIC indicates a better model. Calculated as
      ``AIC = 2k - 2ln(L)`` where k is the number of parameters and L is the likelihood.
      Use for comparing models on the same dataset.

   Anderson-Darling test
      A goodness-of-fit test that gives more weight to the tails of the distribution
      compared to the Kolmogorov-Smirnov test. More sensitive to deviations in the
      tails, making it useful for heavy-tailed data analysis.

   Backend
      An execution engine for running distribution fitting computations. spark-bestfit
      supports three backends: Spark (for clusters), Ray (for ML workflows), and
      Local (for development/testing). See :doc:`backends` for details.

   BIC
      **Bayesian Information Criterion**. Similar to AIC but with a stronger penalty
      for model complexity. Calculated as ``BIC = k*ln(n) - 2ln(L)`` where n is the
      sample size. Tends to select simpler models than AIC for large datasets.

   Bounded fitting
      Distribution fitting where the data is constrained to a specific interval
      [lower_bound, upper_bound]. Uses truncated distributions to ensure samples
      stay within bounds. See :doc:`features/bounded`.

   CDF
      **Cumulative Distribution Function**. The probability that a random variable
      takes a value less than or equal to x: ``F(x) = P(X <= x)``. Ranges from 0 to 1.
      Used in goodness-of-fit tests and probability calculations.

   Confidence interval
      A range of values that contains the true parameter value with a specified
      probability (e.g., 95%). spark-bestfit computes confidence intervals via
      bootstrap resampling.

   Copula
      A function that joins univariate marginal distributions to form a multivariate
      distribution. The Gaussian copula in spark-bestfit preserves correlation
      structure between columns while maintaining each column's fitted marginal
      distribution. See :doc:`features/copula`.

   Goodness-of-fit
      A measure of how well a statistical model fits a set of observations. Common
      metrics include KS statistic, Anderson-Darling statistic, p-value, SSE, AIC,
      and BIC.

   Heavy-tailed distribution
      A distribution with tails that decay more slowly than an exponential
      distribution. Examples include Pareto, Cauchy, and Student's t. Heavy-tailed
      data has high kurtosis and more extreme values than normal data.
      See :doc:`features/heavy-tail`.

   Histogram
      A graphical representation of data distribution showing frequency counts
      across binned intervals. spark-bestfit uses distributed histogram computation
      for large datasets.

   Kolmogorov-Smirnov test
      A nonparametric test that compares a sample distribution to a reference
      distribution (or two samples). The KS statistic measures the maximum distance
      between the empirical CDF and the theoretical CDF. Smaller values indicate
      better fit.

   Kurtosis
      A measure of the "tailedness" of a distribution. High kurtosis (> 3) indicates
      heavy tails and more extreme values. Normal distribution has kurtosis of 3.
      spark-bestfit uses kurtosis to detect heavy-tailed data.

   Lazy metrics
      Deferred computation of expensive goodness-of-fit metrics (KS, AD tests).
      Metrics are computed only when accessed, improving performance when fitting
      many distributions. See :doc:`features/lazy-metrics`.

   Marginal distribution
      The distribution of a single variable in a multivariate context, ignoring
      the other variables. In copula sampling, each column has its own marginal
      distribution that was fit independently.

   Maximum Likelihood Estimation (MLE)
      The default method for estimating distribution parameters by finding the
      values that maximize the probability of observing the data. Works well for
      most distributions but can struggle with heavy-tailed data.

   Maximum Spacing Estimation (MSE)
      An alternative to MLE that maximizes the geometric mean of spacings between
      ordered data points. More robust than MLE for heavy-tailed distributions.
      See :doc:`features/mse-estimation`.

   PDF
      **Probability Density Function**. For continuous distributions, gives the
      relative likelihood of a random variable taking a specific value. The area
      under the PDF curve over an interval gives the probability of falling in
      that interval.

   PMF
      **Probability Mass Function**. For discrete distributions, gives the
      probability that a random variable equals a specific value: ``P(X = x)``.
      Analogous to PDF for continuous distributions.

   PPF
      **Percent Point Function** (also called quantile function or inverse CDF).
      Given a probability p, returns the value x such that ``P(X <= x) = p``.
      Used in sampling to transform uniform random numbers to the target
      distribution.

   P-P plot
      **Probability-Probability plot**. Compares the theoretical CDF against the
      empirical CDF. Points on the diagonal indicate good fit. Deviations show
      systematic differences between the data and fitted distribution.

   p-value
      The probability of obtaining test results at least as extreme as the observed
      results, assuming the null hypothesis is true. In goodness-of-fit testing,
      higher p-values (> 0.05) suggest the data is consistent with the fitted
      distribution.

   Prefiltering
      A performance optimization that eliminates unlikely distribution candidates
      before fitting based on data characteristics (skewness, support, bounds).
      See :doc:`features/prefiltering`.

   Q-Q plot
      **Quantile-Quantile plot**. Compares the quantiles of the data against the
      quantiles of the fitted distribution. Points on the diagonal indicate good
      fit. Deviations in the tails indicate poor tail behavior.

   Sample
      (1) A subset of data drawn from a larger population. (2) To generate random
      values from a fitted distribution.

   Scipy.stats
      The statistics module of the SciPy library, which provides implementations
      of ~90 continuous and 16 discrete probability distributions. spark-bestfit
      uses scipy.stats as its underlying distribution library.

   Skewness
      A measure of asymmetry in a distribution. Positive skewness means a longer
      right tail; negative skewness means a longer left tail. Normal distribution
      has skewness of 0.

   SSE
      **Sum of Squared Errors**. Measures the total squared deviation between
      observed and fitted values. Calculated as ``sum((observed - fitted)^2)``.
      Lower SSE indicates better fit.

   Truncated distribution
      A distribution restricted to a finite interval by conditioning on the
      variable falling within that interval. Used in bounded fitting to ensure
      samples respect known constraints.

   UDF
      **User-Defined Function**. In Spark, a function that can be applied to
      DataFrame columns. spark-bestfit uses UDFs for distributed computation
      of distribution fitting.
