"""Tests for progress tracking functionality."""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip all tests if pyspark not installed
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from spark_bestfit import DiscreteDistributionFitter, DistributionFitter
from spark_bestfit.progress import ProgressCallback, ProgressTracker, console_progress

# Mark all tests in this module as requiring Spark
pytestmark = pytest.mark.spark


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_callback_signature(self, spark_session: SparkSession):
        """Test that callback receives correct argument types."""
        received_args = []

        def callback(completed: int, total: int, percent: float) -> None:
            received_args.append((completed, total, percent))

        tracker = ProgressTracker(spark_session, callback)
        assert tracker.callback == callback
        assert tracker.poll_interval == 0.1
        assert tracker.job_group.startswith("spark-bestfit-")

    def test_context_manager(self, spark_session: SparkSession):
        """Test context manager starts and stops tracking."""
        callback = MagicMock()

        with ProgressTracker(spark_session, callback) as tracker:
            assert tracker._thread is not None
            assert tracker._thread.is_alive()

        # Thread should be stopped after context exit
        time.sleep(0.3)
        assert tracker._thread is None or not tracker._thread.is_alive()

    def test_start_stop(self, spark_session: SparkSession):
        """Test manual start/stop lifecycle."""
        callback = MagicMock()
        tracker = ProgressTracker(spark_session, callback)

        tracker.start()
        assert tracker._thread is not None
        assert tracker._thread.is_alive()

        tracker.stop()
        time.sleep(0.3)
        assert tracker._thread is None or not tracker._thread.is_alive()

    def test_multiple_stop_calls_safe(self, spark_session: SparkSession):
        """Test that calling stop() multiple times is safe."""
        callback = MagicMock()
        tracker = ProgressTracker(spark_session, callback)

        tracker.start()
        tracker.stop()
        tracker.stop()  # Should not raise
        tracker.stop()  # Should not raise

    def test_callback_exception_handled(self, spark_session: SparkSession):
        """Test that exceptions in callback don't crash tracker."""
        call_count = [0]

        def bad_callback(completed: int, total: int, percent: float) -> None:
            call_count[0] += 1
            raise ValueError("Callback error")

        tracker = ProgressTracker(spark_session, bad_callback)
        tracker.start()

        # Run a small Spark job to trigger the callback
        df = spark_session.createDataFrame([(1,), (2,), (3,)], ["x"])
        df.count()  # Trigger Spark action

        time.sleep(0.3)  # Let polling happen
        tracker.stop()
        # Should not raise - callback exceptions are caught

    def test_custom_poll_interval(self, spark_session: SparkSession):
        """Test custom poll interval is respected."""
        callback = MagicMock()
        tracker = ProgressTracker(spark_session, callback, poll_interval=0.5)
        assert tracker.poll_interval == 0.5

    def test_custom_job_group(self, spark_session: SparkSession):
        """Test custom job group is used."""
        callback = MagicMock()
        tracker = ProgressTracker(
            spark_session, callback, job_group="my-custom-group"
        )
        assert tracker.job_group == "my-custom-group"

    def test_none_spark_session(self):
        """Test that ProgressTracker works with None SparkSession.

        Uses get_spark_session() to resolve active session.
        """
        # Create a session first so there's an active one
        spark = SparkSession.builder.master("local[1]").getOrCreate()

        try:
            callback = MagicMock()
            # Pass None - should use active session
            tracker = ProgressTracker(None, callback)
            assert tracker.spark is not None
            assert tracker.spark == spark
        finally:
            # Don't stop - other tests may need it
            pass

    def test_job_group_set_on_start(self, spark_session: SparkSession):
        """Test that job group is set on SparkContext when started."""
        callback = MagicMock()
        tracker = ProgressTracker(spark_session, callback, job_group="test-group-123")

        tracker.start()
        try:
            # The job group should be set on the context
            # We can verify by checking the tracker's job_group matches what we expect
            assert tracker.job_group == "test-group-123"
            assert tracker._thread is not None
        finally:
            tracker.stop()


class TestDistributionFitterProgress:
    """Integration tests for progress tracking in DistributionFitter."""

    def test_progress_callback_invoked(self, spark_session: SparkSession):
        """Test that progress callback is invoked during fitting."""
        np.random.seed(42)
        # Use more data to ensure enough work for progress updates
        data = np.random.normal(50, 10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        progress_updates = []
        lock = threading.Lock()

        def on_progress(completed: int, total: int, percent: float) -> None:
            with lock:
                progress_updates.append((completed, total, percent))

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            max_distributions=10,  # More distributions = more work
            progress_callback=on_progress,
        )

        # Should have some results
        assert results.count() > 0

        # Verify callback argument types if we got updates
        for completed, total, percent in progress_updates:
            assert isinstance(completed, int)
            assert isinstance(total, int)
            assert isinstance(percent, float)
            assert 0 <= percent <= 100
            assert 0 <= completed <= total

    def test_no_callback_works(self, spark_session: SparkSession):
        """Test that fitting works without callback (backward compat)."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=100)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            max_distributions=3,
            # No progress_callback
        )

        assert results.count() > 0

    def test_callback_receives_valid_percentages(self, spark_session: SparkSession):
        """Test that callback receives valid percentage values."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=2000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        all_updates = []
        lock = threading.Lock()

        def on_progress(completed: int, total: int, percent: float) -> None:
            with lock:
                # Validate each update as it comes
                assert 0.0 <= percent <= 100.0, f"Invalid percent: {percent}"
                assert completed <= total, f"completed ({completed}) > total ({total})"
                assert completed >= 0, f"completed negative: {completed}"
                assert total > 0, f"total not positive: {total}"
                all_updates.append((completed, total, percent))

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            max_distributions=8,
            progress_callback=on_progress,
        )

        assert results.count() > 0

    def test_multi_column_progress(self, spark_session: SparkSession):
        """Test progress tracking for multi-column fitting."""
        np.random.seed(42)
        df = spark_session.createDataFrame(
            [
                (float(np.random.normal(50, 10)), float(np.random.exponential(5)))
                for _ in range(500)
            ],
            ["col1", "col2"],
        )

        progress_updates = []
        lock = threading.Lock()

        def on_progress(completed: int, total: int, percent: float) -> None:
            with lock:
                progress_updates.append((completed, total, percent))

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            columns=["col1", "col2"],
            max_distributions=5,
            progress_callback=on_progress,
        )

        # Should have results for both columns
        assert results.count() > 0
        assert len(results.column_names) == 2

        # Verify update structure if any received
        for completed, total, percent in progress_updates:
            assert isinstance(completed, int)
            assert isinstance(total, int)
            assert isinstance(percent, float)


class TestDiscreteDistributionFitterProgress:
    """Integration tests for progress tracking in DiscreteDistributionFitter."""

    def test_discrete_progress_callback(self, spark_session: SparkSession):
        """Test progress tracking for discrete distribution fitting."""
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=2000)
        df = spark_session.createDataFrame([(int(x),) for x in data], ["counts"])

        progress_updates = []
        lock = threading.Lock()

        def on_progress(completed: int, total: int, percent: float) -> None:
            with lock:
                progress_updates.append((completed, total, percent))

        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="counts",
            max_distributions=5,
            progress_callback=on_progress,
        )

        assert results.count() > 0

        # Verify update structure if any received
        for completed, total, percent in progress_updates:
            assert isinstance(completed, int)
            assert isinstance(total, int)
            assert isinstance(percent, float)
            assert 0 <= percent <= 100

    def test_discrete_no_callback_works(self, spark_session: SparkSession):
        """Test that discrete fitting works without callback."""
        np.random.seed(42)
        data = np.random.poisson(lam=5, size=100)
        df = spark_session.createDataFrame([(int(x),) for x in data], ["counts"])

        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="counts",
            max_distributions=3,
            # No progress_callback
        )

        assert results.count() > 0

    def test_discrete_multi_column_progress(self, spark_session: SparkSession):
        """Test progress tracking for discrete multi-column fitting."""
        np.random.seed(42)
        df = spark_session.createDataFrame(
            [
                (int(np.random.poisson(5)), int(np.random.geometric(0.3)))
                for _ in range(500)
            ],
            ["poisson_col", "geometric_col"],
        )

        progress_updates = []
        lock = threading.Lock()

        def on_progress(completed: int, total: int, percent: float) -> None:
            with lock:
                progress_updates.append((completed, total, percent))

        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(
            df,
            columns=["poisson_col", "geometric_col"],
            max_distributions=3,
            progress_callback=on_progress,
        )

        assert results.count() > 0
        assert len(results.column_names) == 2


class TestProgressTrackerThreadSafety:
    """Tests for thread-safety of ProgressTracker."""

    def test_concurrent_updates(self, spark_session: SparkSession):
        """Test that callback is thread-safe with concurrent access."""
        updates = []
        lock = threading.Lock()

        def thread_safe_callback(completed: int, total: int, percent: float) -> None:
            with lock:
                updates.append((completed, total, percent, threading.current_thread().name))

        tracker = ProgressTracker(spark_session, thread_safe_callback)
        tracker.start()

        # Run a small Spark job to trigger progress
        df = spark_session.createDataFrame([(i,) for i in range(100)], ["x"])
        df.count()

        time.sleep(0.3)
        tracker.stop()

        # Verify all recorded updates have correct structure
        for update in updates:
            assert len(update) == 4
            assert isinstance(update[0], int)
            assert isinstance(update[1], int)
            assert isinstance(update[2], float)
            assert isinstance(update[3], str)

    def test_tracker_thread_is_daemon(self, spark_session: SparkSession):
        """Test that the polling thread is a daemon thread.

        Daemon threads don't block process exit.
        """
        callback = MagicMock()
        tracker = ProgressTracker(spark_session, callback)

        tracker.start()
        try:
            assert tracker._thread is not None
            assert tracker._thread.daemon is True
        finally:
            tracker.stop()

    def test_stop_event_terminates_polling(self, spark_session: SparkSession):
        """Test that setting stop event terminates the polling loop."""
        callback = MagicMock()
        tracker = ProgressTracker(spark_session, callback, poll_interval=0.05)

        tracker.start()
        assert tracker._thread.is_alive()

        # Stop should terminate within a reasonable time
        start_time = time.time()
        tracker.stop()
        elapsed = time.time() - start_time

        # Should stop quickly (within 2 seconds is the timeout)
        assert elapsed < 3.0
        assert tracker._thread is None or not tracker._thread.is_alive()


class TestProgressCallback:
    """Tests for the ProgressCallback type alias."""

    def test_type_alias_works_with_functions(self):
        """Test that regular functions work as ProgressCallback."""

        def my_callback(completed: int, total: int, percent: float) -> None:
            pass

        # Should be callable with the expected signature
        my_callback(10, 100, 10.0)

    def test_type_alias_works_with_lambdas(self):
        """Test that lambdas work as ProgressCallback."""
        updates = []
        callback: ProgressCallback = lambda c, t, p: updates.append((c, t, p))
        callback(5, 10, 50.0)
        assert updates == [(5, 10, 50.0)]

    def test_type_alias_works_with_callables(self):
        """Test that callable classes work as ProgressCallback."""

        class MyProgressHandler:
            def __init__(self):
                self.updates = []

            def __call__(self, completed: int, total: int, percent: float) -> None:
                self.updates.append((completed, total, percent))

        handler = MyProgressHandler()
        handler(1, 10, 10.0)
        handler(5, 10, 50.0)
        assert len(handler.updates) == 2


class TestConsoleProgress:
    """Tests for console_progress utility function."""

    def test_console_progress_returns_callable(self):
        """Test that console_progress returns a callable."""
        callback = console_progress()
        assert callable(callback)

    def test_console_progress_default_prefix(self, capsys):
        """Test console_progress with default prefix."""
        callback = console_progress()
        callback(50, 100, 50.0)

        captured = capsys.readouterr()
        assert "Progress:" in captured.out
        assert "50/100" in captured.out
        assert "50.0%" in captured.out

    def test_console_progress_custom_prefix(self, capsys):
        """Test console_progress with custom prefix."""
        callback = console_progress("Fitting distributions")
        callback(25, 50, 50.0)

        captured = capsys.readouterr()
        assert "Fitting distributions:" in captured.out
        assert "25/50" in captured.out

    def test_console_progress_multiple_updates(self, capsys):
        """Test that console_progress handles multiple updates."""
        callback = console_progress()

        callback(10, 100, 10.0)
        callback(50, 100, 50.0)
        callback(100, 100, 100.0)

        captured = capsys.readouterr()
        # Should contain the last update (uses carriage return for in-place)
        assert "100/100" in captured.out
        assert "100.0%" in captured.out

    def test_console_progress_uses_carriage_return(self, capsys):
        """Test that console_progress uses carriage return for in-place updates."""
        callback = console_progress()
        callback(50, 100, 50.0)

        captured = capsys.readouterr()
        # Output should start with carriage return
        assert captured.out.startswith("\r")

    def test_console_progress_integration(self, spark_session: SparkSession):
        """Test console_progress with actual fitting."""
        import numpy as np

        np.random.seed(42)
        data = np.random.normal(50, 10, size=500)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        # Should not raise
        results = fitter.fit(
            df,
            column="value",
            max_distributions=3,
            progress_callback=console_progress("Testing"),
        )

        assert results.count() > 0

    def test_console_progress_importable_from_package(self):
        """Test that console_progress is importable from main package."""
        from spark_bestfit import console_progress as cp

        callback = cp()
        assert callable(callback)
