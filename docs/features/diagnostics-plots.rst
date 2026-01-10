Diagnostics Plots
=================

After fitting a distribution, you can assess the quality of the fit using
diagnostic plots. spark-bestfit provides a comprehensive ``diagnostics()``
method that creates a 2x2 panel of diagnostic visualizations.

Quick Start
-----------

Generate a complete diagnostic panel from a fitted distribution:

.. code-block:: python

    from spark_bestfit import DistributionFitter

    # Fit distribution
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value")
    best = results.best(n=1)[0]

    # Get the sample data as numpy array
    data = df.select("value").toPandas()["value"].values

    # Generate diagnostic plots
    fig, axes = best.diagnostics(data, title="Distribution Fit Diagnostics")

The result is a 2x2 matplotlib figure with four diagnostic plots:

.. code-block:: text

    +-------------------+-------------------+
    |      Q-Q Plot     |      P-P Plot     |
    +-------------------+-------------------+
    | Residual Histogram|   CDF Comparison  |
    +-------------------+-------------------+

Diagnostic Plot Types
---------------------

Q-Q Plot (Quantile-Quantile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Q-Q plot compares sample quantiles against theoretical quantiles from
the fitted distribution. Points falling along the diagonal reference line
indicate a good fit.

- **Use for**: Detecting deviations in the tails of the distribution
- **Good fit**: Points closely follow the y=x line
- **Heavy tails**: Points curve away from the line at extremes
- **Light tails**: Points curve toward the line at extremes

P-P Plot (Probability-Probability)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The P-P plot compares the empirical cumulative distribution function (CDF)
against the theoretical CDF. It is particularly sensitive to deviations in
the center of the distribution.

- **Use for**: Assessing fit quality in the center of the distribution
- **Good fit**: Points closely follow the y=x line
- **Bounded axes**: Always [0, 1] for probabilities

Residual Histogram
~~~~~~~~~~~~~~~~~~

Shows the distribution of residuals (observed density - expected density).
A good fit should have residuals centered around zero.

- **Use for**: Identifying systematic bias in the fit
- **Good fit**: Histogram centered at zero, symmetric
- **Metrics shown**: Mean and standard deviation of residuals

CDF Comparison
~~~~~~~~~~~~~~

Overlays the empirical step-function CDF on top of the smooth theoretical
CDF. Visual alignment indicates goodness of fit.

- **Use for**: Direct visual comparison of distributions
- **Good fit**: Step function closely follows smooth curve
- **Shows**: KS statistic and p-value when available

API: diagnostics() Method
-------------------------

The ``diagnostics()`` method is available on ``DistributionFitResult`` objects:

.. code-block:: python

    result.diagnostics(
        data,                      # Sample data (numpy array)
        y_hist=None,               # Optional: pre-computed histogram density
        x_hist=None,               # Optional: pre-computed histogram bin centers
        bins=50,                   # Number of histogram bins
        title="",                  # Overall figure title
        figsize=(14, 12),          # Figure size (width, height)
        dpi=100,                   # Dots per inch for saved figures
        title_fontsize=16,         # Main title font size
        subplot_title_fontsize=12, # Subplot title font size
        label_fontsize=10,         # Axis label font size
        grid_alpha=0.3,            # Grid transparency
        save_path=None,            # Optional path to save figure
        save_format="png",         # Save format (png, pdf, svg)
    )

Returns a tuple of ``(figure, axes)`` where ``axes`` is a 2x2 numpy array
of matplotlib Axes objects.

Example Usage
-------------

Basic Diagnostics
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spark_bestfit import DistributionFitter
    import matplotlib.pyplot as plt

    # Fit and get best distribution
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, "value")
    best = results.best(n=1)[0]

    # Get data for plotting
    data = df.select("value").toPandas()["value"].values

    # Generate diagnostics
    fig, axes = best.diagnostics(data)
    plt.show()

With Pre-computed Histogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    # Pre-compute histogram (useful when reusing across multiple plots)
    y_hist, x_edges = np.histogram(data, bins=50, density=True)
    x_hist = (x_edges[:-1] + x_edges[1:]) / 2

    # Use pre-computed histogram
    fig, axes = best.diagnostics(
        data,
        y_hist=y_hist,
        x_hist=x_hist,
        title="Fit Quality Assessment"
    )

Saving to File
~~~~~~~~~~~~~~

.. code-block:: python

    # Save as PNG
    fig, axes = best.diagnostics(
        data,
        title="Model Diagnostics",
        save_path="diagnostics.png",
        dpi=300
    )

    # Save as PDF for publications
    fig, axes = best.diagnostics(
        data,
        save_path="diagnostics.pdf",
        save_format="pdf"
    )

Comparing Multiple Fits
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get top 3 distributions
    top_3 = results.best(n=3)

    # Create diagnostics for each
    for i, result in enumerate(top_3):
        fig, axes = result.diagnostics(
            data,
            title=f"Rank {i+1}: {result.distribution}",
            save_path=f"diagnostics_{i+1}.png"
        )
        plt.close(fig)

Individual Plot Functions
-------------------------

For more control, individual plotting functions are available:

.. code-block:: python

    from spark_bestfit.plotting import (
        plot_qq,
        plot_pp,
        plot_residual_histogram,
        plot_cdf_comparison,
    )

    # Q-Q plot only
    fig, ax = plot_qq(result, data, title="Q-Q Plot")

    # P-P plot only
    fig, ax = plot_pp(result, data, title="P-P Plot")

    # Residual histogram only
    fig, ax = plot_residual_histogram(result, y_hist, x_hist)

    # CDF comparison only
    fig, ax = plot_cdf_comparison(result, data)

Each function accepts extensive customization parameters for colors, fonts,
markers, and line styles. See the API reference for full details.

Interpreting Results
--------------------

Good Fit Indicators
~~~~~~~~~~~~~~~~~~~

- Q-Q/P-P plots: Points closely follow the diagonal line
- Residual histogram: Centered at zero, symmetric, small standard deviation
- CDF comparison: Empirical CDF closely tracks theoretical CDF
- KS p-value: > 0.05 (though this is only a rough guideline)

Poor Fit Indicators
~~~~~~~~~~~~~~~~~~~

- Q-Q plot: Systematic curvature, especially in tails
- P-P plot: S-shaped deviation from diagonal
- Residual histogram: Mean far from zero, skewed distribution
- CDF comparison: Visible gaps between empirical and theoretical CDFs

API Reference
-------------

See :meth:`spark_bestfit.results.DistributionFitResult.diagnostics` for
full API documentation.

See also:

- :func:`spark_bestfit.plotting.plot_qq`
- :func:`spark_bestfit.plotting.plot_pp`
- :func:`spark_bestfit.plotting.plot_residual_histogram`
- :func:`spark_bestfit.plotting.plot_cdf_comparison`
- :func:`spark_bestfit.plotting.plot_diagnostics`
