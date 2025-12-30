Progress Tracking
=================

spark-bestfit v1.2.0 introduces progress tracking for distribution fitting operations.
This allows you to monitor long-running jobs and provide feedback to users.

Quick Start
-----------

The easiest way to enable progress tracking is with the built-in ``console_progress()`` utility:

.. code-block:: python

    from spark_bestfit import DistributionFitter
    from spark_bestfit.progress import console_progress

    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value", progress_callback=console_progress())
    print()  # Newline after progress

This displays progress like::

    Progress: 45/100 tasks (45.0%)

You can customize the prefix:

.. code-block:: python

    results = fitter.fit(df, column="value", progress_callback=console_progress("Fitting distributions"))
    # Output: Fitting distributions: 45/100 tasks (45.0%)

Custom Callbacks
----------------

For full control, pass any function matching the ``ProgressCallback`` signature:

.. code-block:: python

    from spark_bestfit import DistributionFitter

    def on_progress(completed: int, total: int, percent: float) -> None:
        print(f"\rFitting: {completed}/{total} ({percent:.1f}%)", end="", flush=True)

    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value", progress_callback=on_progress)
    print()  # Newline after progress

The callback receives three arguments:

- ``completed``: Number of tasks completed so far
- ``total``: Total number of tasks in the job
- ``percent``: Percentage complete (0.0 to 100.0)

With tqdm
---------

Integration with tqdm for progress bars:

.. code-block:: python

    from tqdm import tqdm
    from spark_bestfit import DistributionFitter

    pbar = None

    def tqdm_callback(completed: int, total: int, percent: float) -> None:
        global pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Fitting distributions")
        pbar.n = completed
        pbar.refresh()

    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value", progress_callback=tqdm_callback)

    if pbar:
        pbar.close()

Discrete Distribution Fitting
-----------------------------

Progress tracking works the same way for discrete distributions:

.. code-block:: python

    from spark_bestfit import DiscreteDistributionFitter

    fitter = DiscreteDistributionFitter(spark)
    results = fitter.fit(
        df,
        column="counts",
        progress_callback=on_progress
    )

Multi-Column Fitting
--------------------

When fitting multiple columns, progress reflects aggregate completion
across all columns:

.. code-block:: python

    results = fitter.fit(
        df,
        columns=["col1", "col2", "col3"],
        progress_callback=on_progress,
    )
    # Progress shows total tasks across all 3 columns

Thread Safety
-------------

.. warning::

    The callback is invoked from a background thread. Ensure your callback
    implementation is thread-safe. Avoid modifying shared state without
    proper synchronization.

For thread-safe progress tracking:

.. code-block:: python

    import threading

    class ThreadSafeProgress:
        def __init__(self):
            self.lock = threading.Lock()
            self.last_percent = 0.0

        def __call__(self, completed: int, total: int, percent: float) -> None:
            with self.lock:
                if percent - self.last_percent >= 5.0:  # Update every 5%
                    print(f"Progress: {percent:.1f}%")
                    self.last_percent = percent

    progress = ThreadSafeProgress()
    results = fitter.fit(df, column="value", progress_callback=progress)

How It Works
------------

Progress tracking uses Spark's built-in StatusTracker API:

1. When a ``progress_callback`` is provided, a ``ProgressTracker`` is created
2. The tracker sets a unique job group on SparkContext
3. A background thread polls Spark's StatusTracker every 100ms
4. When stage/task progress changes, the callback is invoked
5. After fitting completes, the tracker is automatically stopped

This approach:

- Has minimal overhead (~0.1% increase in runtime)
- Works on all Spark environments (local, YARN, Kubernetes, Databricks)
- Provides partition-level granularity (~16 updates for typical jobs)
- Does not require any changes to the Spark job itself

Advanced Usage: ProgressTracker
-------------------------------

For more control, you can use ``ProgressTracker`` directly:

.. code-block:: python

    from spark_bestfit import ProgressTracker

    def on_progress(completed: int, total: int, percent: float) -> None:
        print(f"Progress: {percent:.1f}%")

    # Using context manager
    with ProgressTracker(spark, on_progress) as tracker:
        # Any Spark operations here will be tracked
        results = fitter.fit(df, column="value")

    # Or manual start/stop
    tracker = ProgressTracker(spark, on_progress, poll_interval=0.5)
    tracker.start()
    try:
        results = fitter.fit(df, column="value")
    finally:
        tracker.stop()

``ProgressTracker`` parameters:

- ``spark``: SparkSession instance (or None to use active session)
- ``callback``: Progress callback function
- ``poll_interval``: Seconds between status checks (default: 0.1)
- ``job_group``: Custom job group identifier (auto-generated if None)

Performance Notes
-----------------

- Progress tracking adds minimal overhead (~0.1% increase in runtime)
- The polling interval of 100ms provides a good balance between responsiveness and overhead
- No impact on Spark job execution - tracking is purely observational
- Works with all Spark cluster managers (standalone, YARN, Kubernetes, Mesos)

Understanding Progress Output
-----------------------------

Progress values may appear to fluctuate during fitting::

    Progress: 34/85 tasks (40.0%)
    Progress: 65/156 tasks (41.7%)
    Progress: 115/216 tasks (53.2%)

This is expected behavior:

- **Total increases**: Each distribution fit triggers Spark stages. As new stages
  start, the total task count grows (85 → 156 → 216 in the example above).

- **Percentage can decrease**: When a new stage starts, the denominator increases
  before its tasks complete, temporarily lowering the percentage.

- **Final may not reach 100%**: The job may complete between polling intervals (100ms),
  so the last captured progress might be less than 100%.

The key observation is that progress generally trends upward, and the job completes
successfully. For long-running fits (many distributions or large datasets), you will
see many incremental updates as stages complete.
