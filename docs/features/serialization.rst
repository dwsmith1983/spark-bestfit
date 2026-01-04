Serialization
=============

spark-bestfit supports serialization for saving and loading fitted distributions.
This allows you to persist fitted models to disk and reload them later for inference,
without needing to re-fit the distributions.

Quick Start
-----------

Save and load a fitted distribution:

.. code-block:: python

    from spark_bestfit import DistributionFitter, DistributionFitResult

    # Fit distributions
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value")
    best = results.best(n=1)[0]

    # Save to JSON (default)
    best.save("model.json")

    # Load and use
    loaded = DistributionFitResult.load("model.json")
    samples = loaded.sample(size=1000)

Supported Formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Format
     - Use Case
     - Extension
   * - **JSON**
     - Human-readable, version-safe, debuggable. Recommended for most use cases.
     - ``.json``
   * - **Pickle**
     - Binary format for Python-only workflows. Faster but not human-readable.
     - ``.pkl``, ``.pickle``

The format is auto-detected from the file extension, or can be specified explicitly:

.. code-block:: python

    # Auto-detect from extension
    best.save("model.json")      # JSON
    best.save("model.pkl")       # Pickle

    # Explicit format (overrides extension)
    best.save("model.dat", format="json")
    best.save("model.dat", format="pickle")

JSON Schema
-----------

The JSON format includes metadata for versioning and debugging:

.. code-block:: json

    {
      "schema_version": "1.0",
      "spark_bestfit_version": "2.0.0",
      "created_at": "2026-01-04T15:30:00.123456+00:00",
      "distribution": "gamma",
      "parameters": [2.0, 0.0, 5.0],
      "column_name": "response_time",
      "metrics": {
        "sse": 0.003,
        "aic": 1400.0,
        "bic": 1430.0,
        "ks_statistic": 0.020,
        "pvalue": 0.95,
        "ad_statistic": 0.40,
        "ad_pvalue": null
      },
      "data_summary": {
        "sample_size": 1000000.0,
        "min": 0.5,
        "max": 245.3,
        "mean": 10.2,
        "std": 8.7
      }
    }

The ``data_summary`` field provides lightweight provenance tracking - it captures basic
statistics about the data used for fitting, without storing the actual data.

Compact JSON
~~~~~~~~~~~~

For smaller file sizes, you can disable indentation:

.. code-block:: python

    # Compact (single line)
    best.save("model.json", indent=None)

    # Custom indentation (default is 2)
    best.save("model.json", indent=4)

Using Loaded Results
--------------------

Loaded results are fully functional ``DistributionFitResult`` objects:

.. code-block:: python

    loaded = DistributionFitResult.load("model.json")

    # Generate samples
    samples = loaded.sample(size=10000, random_state=42)

    # Evaluate PDF/CDF
    import numpy as np
    x = np.linspace(0, 50, 100)
    pdf_values = loaded.pdf(x)
    cdf_values = loaded.cdf(x)

    # Access all metrics
    print(f"Distribution: {loaded.distribution}")
    print(f"Parameters: {loaded.parameters}")
    print(f"K-S statistic: {loaded.ks_statistic}")
    print(f"p-value: {loaded.pvalue}")

    # Access data summary (if available)
    if loaded.data_summary:
        print(f"Original sample size: {loaded.data_summary['sample_size']}")

Data Summary
------------

When fitting distributions with ``DistributionFitter`` or ``DiscreteDistributionFitter``,
the ``data_summary`` field is automatically populated with statistics from the fitted data:

- ``sample_size``: Number of data points
- ``min``: Minimum value
- ``max``: Maximum value
- ``mean``: Mean value
- ``std``: Standard deviation

This provides useful context without requiring full data versioning:

.. code-block:: python

    # Fit and save
    results = fitter.fit(df, column="response_time")
    best = results.best(n=1)[0]
    best.save("model.json")

    # Later: inspect data summary
    loaded = DistributionFitResult.load("model.json")
    summary = loaded.data_summary

    if summary:
        print(f"Model was fit on {summary['sample_size']:.0f} samples")
        print(f"Data range: [{summary['min']:.2f}, {summary['max']:.2f}]")
        print(f"Mean: {summary['mean']:.2f}, Std: {summary['std']:.2f}")

.. note::

    ``data_summary`` may be ``None`` for results created manually or loaded from
    older versions. Always check before accessing.

Creating Results Manually
-------------------------

You can create ``DistributionFitResult`` objects manually for testing or for
distributions fit outside spark-bestfit:

.. code-block:: python

    from spark_bestfit import DistributionFitResult

    # Create from known parameters
    result = DistributionFitResult(
        distribution="gamma",
        parameters=[2.0, 0.0, 5.0],
        sse=0.003,
        aic=1400.0,
        bic=1430.0,
        ks_statistic=0.020,
        pvalue=0.95,
    )

    # Save and load works the same
    result.save("manual_fit.json")
    loaded = DistributionFitResult.load("manual_fit.json")

Error Handling
--------------

The ``SerializationError`` exception is raised for serialization-related errors:

.. code-block:: python

    from spark_bestfit import DistributionFitResult, SerializationError

    try:
        loaded = DistributionFitResult.load("model.json")
    except FileNotFoundError:
        print("File not found")
    except SerializationError as e:
        print(f"Serialization error: {e}")

Common errors include:

- **Missing required fields**: JSON is missing ``distribution`` or ``parameters``
- **Unknown distribution**: The distribution name is not recognized by scipy.stats
- **Invalid JSON**: The file contains malformed JSON
- **Unknown format**: File extension is not ``.json``, ``.pkl``, or ``.pickle``

Workflow Example
----------------

A typical workflow for model persistence:

.. code-block:: python

    from spark_bestfit import DistributionFitter, DistributionFitResult
    from pathlib import Path

    # --- Training ---
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="latency")

    # Get top 3 fits
    top_fits = results.best(n=3)
    print("Top distributions:")
    for fit in top_fits:
        print(f"  {fit.distribution}: KS={fit.ks_statistic:.4f}")

    # Save the best
    best = top_fits[0]
    best.save("models/latency_model.json")

    # --- Later: Inference ---
    model = DistributionFitResult.load("models/latency_model.json")

    # Generate synthetic data
    synthetic_samples = model.sample(size=100000, random_state=42)

    # Calculate percentiles
    import numpy as np
    p95 = model.ppf(0.95)
    p99 = model.ppf(0.99)
    print(f"P95: {p95:.2f}, P99: {p99:.2f}")

    # Probability of exceeding threshold
    prob_slow = 1 - model.cdf(100)  # P(latency > 100ms)
    print(f"Probability of >100ms: {prob_slow:.2%}")

Multi-Distribution Persistence
------------------------------

To save multiple distributions from the same fitting session:

.. code-block:: python

    import json
    from pathlib import Path

    # Fit and get all good results
    results = fitter.fit(df, column="value")
    good_fits = results.filter(pvalue_threshold=0.05)

    # Save each to a separate file
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    manifest = []
    for fit in good_fits.best(n=10):
        filename = f"{fit.distribution}.json"
        fit.save(models_dir / filename)
        manifest.append({
            "distribution": fit.distribution,
            "file": filename,
            "ks_statistic": fit.ks_statistic,
            "pvalue": fit.pvalue,
        })

    # Save manifest
    with open(models_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

API Reference
-------------

See :meth:`spark_bestfit.results.DistributionFitResult.save` and
:meth:`spark_bestfit.results.DistributionFitResult.load` for full API documentation.
