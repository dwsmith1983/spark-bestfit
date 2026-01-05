"""Progress tracking for Spark distribution fitting jobs.

This module provides a StatusTracker-based approach to monitor Spark job
progress during distribution fitting. It uses Spark's built-in job group
and status tracking APIs to poll for progress updates.

Example:
    >>> def on_progress(completed: int, total: int, percent: float) -> None:
    ...     print(f"Progress: {completed}/{total} ({percent:.1f}%)")
    >>>
    >>> fitter = DistributionFitter(spark)
    >>> results = fitter.fit(df, column="value", progress_callback=on_progress)
"""

import logging
import threading
import time
from typing import Callable, Optional

# PySpark is optional - only import if available
try:
    from pyspark.sql import SparkSession

    _PYSPARK_AVAILABLE = True
except ImportError:
    SparkSession = None  # type: ignore[assignment,misc]
    _PYSPARK_AVAILABLE = False

from spark_bestfit.utils import get_spark_session

logger = logging.getLogger(__name__)

# Type alias for progress callback
# Args: (completed_tasks, total_tasks, percent_complete)
ProgressCallback = Callable[[int, int, float], None]


class ProgressTracker:
    """Track progress of Spark fitting jobs and invoke callbacks.

    Uses Spark's StatusTracker API to poll job/stage progress
    and invokes user callback at partition-level granularity.

    Thread Safety:
        The callback is invoked from a background thread. Users must
        ensure their callback implementation is thread-safe.

    Example:
        >>> def on_progress(completed: int, total: int, percent: float) -> None:
        ...     print(f"Progress: {completed}/{total} ({percent:.1f}%)")
        >>>
        >>> # Using context manager
        >>> with ProgressTracker(spark, on_progress) as tracker:
        ...     results = fitter.fit(df, column="value")
        >>>
        >>> # Or manual start/stop
        >>> tracker = ProgressTracker(spark, on_progress)
        >>> tracker.start()
        >>> try:
        ...     results = fitter.fit(df, column="value")
        ... finally:
        ...     tracker.stop()

    Attributes:
        spark: SparkSession instance (resolved from active session if None provided)
        callback: Function called on progress updates
        poll_interval: Seconds between status checks (default 0.1)
        job_group: Unique identifier for tracking this job
    """

    def __init__(
        self,
        spark: Optional[SparkSession],
        callback: ProgressCallback,
        poll_interval: float = 0.1,
        job_group: Optional[str] = None,
    ):
        """Initialize ProgressTracker.

        Args:
            spark: SparkSession instance. If None, uses active session.
            callback: Function to call on progress updates.
                Signature: (completed: int, total: int, percent: float) -> None
            poll_interval: Seconds between status checks (default 0.1 = 100ms)
            job_group: Optional job group identifier. Auto-generated if None.
        """
        self.spark: SparkSession = get_spark_session(spark)
        self.callback = callback
        self.poll_interval = poll_interval
        self.job_group = job_group or f"spark-bestfit-{id(self)}-{time.time_ns()}"

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Progress state
        self._completed_tasks = 0
        self._total_tasks = 0

    def start(self) -> None:
        """Start progress tracking.

        Sets the job group on SparkContext and starts a background
        thread to poll for progress updates.
        """
        # Set job group for this fitting operation
        self.spark.sparkContext.setJobGroup(
            self.job_group,
            "spark-bestfit distribution fitting",
            interruptOnCancel=True,
        )

        self._stop_event.clear()
        self._completed_tasks = 0
        self._total_tasks = 0

        self._thread = threading.Thread(
            target=self._poll_loop,
            name=f"ProgressTracker-{self.job_group[:20]}",
            daemon=True,
        )
        self._thread.start()
        logger.debug(f"Started progress tracking for job group: {self.job_group}")

    def stop(self) -> None:
        """Stop progress tracking and cleanup.

        Stops the polling thread and clears the job group.
        Safe to call multiple times.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Clear job group
        try:
            self.spark.sparkContext.clearJobGroup()
        except Exception:
            pass  # SparkContext may be stopped

        logger.debug(f"Stopped progress tracking for job group: {self.job_group}")

    def _poll_loop(self) -> None:
        """Background thread that polls StatusTracker for progress."""
        sc = self.spark.sparkContext
        status_tracker = sc.statusTracker()

        while not self._stop_event.is_set():
            try:
                self._update_progress(status_tracker)
            except Exception as e:
                # Log but don't crash - job may have completed
                logger.debug(f"Progress polling error (may be normal): {e}")

            self._stop_event.wait(timeout=self.poll_interval)

    def _update_progress(self, status_tracker) -> None:
        """Poll StatusTracker and invoke callback if progress changed."""
        # Get jobs for our job group
        job_ids = status_tracker.getJobIdsForGroup(self.job_group)

        if not job_ids:
            return

        total = 0
        completed = 0

        for job_id in job_ids:
            job_info = status_tracker.getJobInfo(job_id)
            if job_info is None:
                continue

            for stage_id in job_info.stageIds:
                stage_info = status_tracker.getStageInfo(stage_id)
                if stage_info is not None:
                    total += stage_info.numTasks
                    completed += stage_info.numCompletedTasks

        # Update state and invoke callback if changed
        with self._lock:
            if total > 0 and (completed != self._completed_tasks or total != self._total_tasks):
                self._completed_tasks = completed
                self._total_tasks = total
                percent = (completed / total) * 100.0

                # Invoke callback with error handling
                try:
                    self.callback(completed, total, percent)
                except Exception as e:
                    logger.warning(f"Progress callback raised exception: {e}")

    def __enter__(self) -> "ProgressTracker":
        """Context manager entry - starts tracking."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops tracking."""
        self.stop()


def console_progress(prefix: str = "Progress") -> ProgressCallback:
    """Create a simple console progress callback.

    Returns a callback that prints progress updates to stdout with
    in-place updates using carriage return.

    Args:
        prefix: Text to display before progress info. Defaults to "Progress".

    Returns:
        A progress callback function suitable for ``fit(progress_callback=...)``.

    Example:
        >>> from spark_bestfit import DistributionFitter
        >>> from spark_bestfit.progress import console_progress
        >>>
        >>> fitter = DistributionFitter(spark)
        >>> results = fitter.fit(df, column="value", progress_callback=console_progress())
        Progress: 45/100 tasks (45.0%)
        >>>
        >>> # With custom prefix
        >>> results = fitter.fit(df, column="value", progress_callback=console_progress("Fitting"))
        Fitting: 45/100 tasks (45.0%)

    Note:
        Progress values may fluctuate during fitting because Spark jobs
        consist of multiple stages. As new stages start, the total task
        count increases, which can temporarily decrease the percentage.
        This is normal behavior - progress generally trends upward.
    """
    import sys

    def callback(completed: int, total: int, percent: float) -> None:
        # Use carriage return for in-place updates, flush to ensure display
        sys.stdout.write(f"\r{prefix}: {completed}/{total} tasks ({percent:.1f}%)")
        sys.stdout.flush()

    return callback
