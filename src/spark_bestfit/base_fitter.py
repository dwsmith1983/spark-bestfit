"""Base class for distribution fitters - eliminates code duplication."""

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from scipy import stats

from spark_bestfit.config import SamplingMode

# PySpark is optional - only import if available
try:
    import pyspark.sql.functions as F
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql.types import NumericType

    _PYSPARK_AVAILABLE = True
except ImportError:
    F = None  # type: ignore[assignment]
    DataFrame = None  # type: ignore[assignment,misc]
    SparkSession = None  # type: ignore[assignment,misc]
    NumericType = None  # type: ignore[assignment,misc]
    _PYSPARK_AVAILABLE = False

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend

# Import get_spark_session only if PySpark is available
if _PYSPARK_AVAILABLE:
    from spark_bestfit.utils import get_spark_session

logger = logging.getLogger(__name__)


class BaseFitter(ABC):
    """Abstract base class for distribution fitters.

    Provides common functionality for both continuous and discrete distribution
    fitting, including input validation, bounds handling, and sampling logic.

    Subclasses must define:
        - _registry_class: The distribution registry class to use
        - _default_exclusions: Default distributions to exclude
        - fit(): The main fitting method with distribution-specific logic
    """

    # Class attributes - must be set by subclasses
    _registry_class: Type[Any]  # DistributionRegistry or DiscreteDistributionRegistry
    _default_exclusions: Tuple[str, ...]  # Default exclusions for this fitter type

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        excluded_distributions: Optional[Tuple[str, ...]] = None,
        random_seed: int = 42,
        backend: Optional["ExecutionBackend"] = None,
    ):
        """Initialize the distribution fitter.

        Args:
            spark: SparkSession. If None, uses the active session.
                Ignored if ``backend`` is provided.
            excluded_distributions: Distributions to exclude from fitting.
                Defaults to the fitter's _default_exclusions.
                Pass an empty tuple ``()`` to include ALL distributions.
            random_seed: Random seed for reproducible sampling.
            backend: Optional execution backend. If None, creates a
                SparkBackend from the spark session.

        Raises:
            RuntimeError: If no SparkSession provided and no active session exists
        """
        # Initialize backend (lazy import to avoid circular dependency)
        if backend is not None:
            self._backend = backend
            # Extract SparkSession from SparkBackend if available
            if hasattr(backend, "spark"):
                self.spark = backend.spark
            else:
                # For non-Spark backends (LocalBackend, RayBackend), no SparkSession needed
                self.spark = None
        else:
            # Default to SparkBackend - requires PySpark
            if not _PYSPARK_AVAILABLE:
                raise ImportError(
                    "PySpark is required when no backend is specified. "
                    "Install with: pip install spark-bestfit[spark]\n"
                    "Or use a non-Spark backend: LocalBackend() or RayBackend()"
                )
            self.spark = get_spark_session(spark)
            # Lazy import to avoid circular dependency
            from spark_bestfit.backends.spark import SparkBackend

            self._backend = SparkBackend(self.spark)

        self.excluded_distributions = (
            excluded_distributions if excluded_distributions is not None else self._default_exclusions
        )
        self.random_seed = random_seed

        # When excluded_distributions=() is explicitly passed, disable registry's
        # default exclusions so ALL distributions are available
        if excluded_distributions == ():
            self._registry = self._registry_class(custom_exclusions=set())
        else:
            self._registry = self._registry_class()

    # Note: fit() is intentionally NOT abstract - subclasses have different
    # signatures (continuous has bins/use_rice_rule, discrete doesn't).
    # Using duck typing here instead of strict ABC enforcement.

    @staticmethod
    def _validate_bounds(
        lower_bound: Optional[Union[float, Dict[str, float]]],
        upper_bound: Optional[Union[float, Dict[str, float]]],
        target_columns: List[str],
    ) -> None:
        """Validate bounds parameters.

        Args:
            lower_bound: Scalar or dict of lower bounds
            upper_bound: Scalar or dict of upper bounds
            target_columns: List of columns being fitted

        Raises:
            ValueError: If bounds are invalid (lower >= upper, unknown columns in dict)
        """
        # Validate scalar bounds
        if isinstance(lower_bound, (int, float)) and isinstance(upper_bound, (int, float)):
            if lower_bound >= upper_bound:
                raise ValueError(f"lower_bound ({lower_bound}) must be less than upper_bound ({upper_bound})")
            return

        # Validate dict bounds - check for unknown columns
        if isinstance(lower_bound, dict):
            unknown = set(lower_bound.keys()) - set(target_columns)
            if unknown:
                raise ValueError(f"lower_bound contains unknown columns: {unknown}. Valid columns: {target_columns}")

        if isinstance(upper_bound, dict):
            unknown = set(upper_bound.keys()) - set(target_columns)
            if unknown:
                raise ValueError(f"upper_bound contains unknown columns: {unknown}. Valid columns: {target_columns}")

        # Validate that lower < upper for each column where both are specified
        lower_dict = lower_bound if isinstance(lower_bound, dict) else {}
        upper_dict = upper_bound if isinstance(upper_bound, dict) else {}

        for col in target_columns:
            col_lower = lower_dict.get(col) if isinstance(lower_bound, dict) else lower_bound
            col_upper = upper_dict.get(col) if isinstance(upper_bound, dict) else upper_bound
            if col_lower is not None and col_upper is not None:
                if col_lower >= col_upper:
                    raise ValueError(
                        f"lower_bound ({col_lower}) must be less than upper_bound ({col_upper}) for column '{col}'"
                    )

    @staticmethod
    def _resolve_bounds(
        df: DataFrame,
        target_columns: List[str],
        lower_bound: Optional[Union[float, Dict[str, float]]],
        upper_bound: Optional[Union[float, Dict[str, float]]],
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Resolve bounds to per-column dict, auto-detecting from data where needed.

        Args:
            df: DataFrame containing data
            target_columns: List of columns being fitted
            lower_bound: Scalar, dict, or None
            upper_bound: Scalar, dict, or None

        Returns:
            Dict mapping column name to (lower, upper) tuple
        """
        # Determine which columns need auto-detection
        lower_dict = lower_bound if isinstance(lower_bound, dict) else {}
        upper_dict = upper_bound if isinstance(upper_bound, dict) else {}

        cols_need_lower = [
            col for col in target_columns if not isinstance(lower_bound, (int, float)) and col not in lower_dict
        ]
        cols_need_upper = [
            col for col in target_columns if not isinstance(upper_bound, (int, float)) and col not in upper_dict
        ]

        # Execute aggregation for auto-detection - handle different DataFrame types
        auto_bounds: Dict[str, float] = {}
        cols_to_check = list(set(cols_need_lower + cols_need_upper))

        if cols_to_check:
            if hasattr(df, "sparkSession"):
                # Spark DataFrame
                agg_exprs = []
                for col in cols_need_lower:
                    agg_exprs.append(F.min(col).alias(f"min_{col}"))
                for col in cols_need_upper:
                    agg_exprs.append(F.max(col).alias(f"max_{col}"))
                bounds_row = df.agg(*agg_exprs).first()
                for col in cols_need_lower:
                    auto_bounds[f"min_{col}"] = float(bounds_row[f"min_{col}"])
                for col in cols_need_upper:
                    auto_bounds[f"max_{col}"] = float(bounds_row[f"max_{col}"])
            elif hasattr(df, "select_columns") and hasattr(df, "min"):
                # Ray Dataset
                for col in cols_need_lower:
                    auto_bounds[f"min_{col}"] = float(df.min(col))
                for col in cols_need_upper:
                    auto_bounds[f"max_{col}"] = float(df.max(col))
            else:
                # pandas DataFrame
                for col in cols_need_lower:
                    auto_bounds[f"min_{col}"] = float(df[col].min())
                for col in cols_need_upper:
                    auto_bounds[f"max_{col}"] = float(df[col].max())

        # Build final per-column bounds dict
        result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for col in target_columns:
            # Determine lower bound for this column
            if isinstance(lower_bound, (int, float)):
                col_lower = float(lower_bound)
            elif isinstance(lower_bound, dict) and col in lower_bound:
                col_lower = float(lower_bound[col])
            else:
                col_lower = auto_bounds.get(f"min_{col}")

            # Determine upper bound for this column
            if isinstance(upper_bound, (int, float)):
                col_upper = float(upper_bound)
            elif isinstance(upper_bound, dict) and col in upper_bound:
                col_upper = float(upper_bound[col])
            else:
                col_upper = auto_bounds.get(f"max_{col}")

            result[col] = (col_lower, col_upper)
            logger.info(f"Bounded fitting for '{col}': bounds=[{col_lower}, {col_upper}]")

        return result

    def _apply_sampling(
        self,
        df: DataFrame,
        row_count: int,
        enable_sampling: bool,
        sample_fraction: Optional[float],
        max_sample_size: int,
        sample_threshold: int,
        column: Optional[str] = None,
        adaptive_sampling: bool = True,
        sampling_mode: SamplingMode = SamplingMode.AUTO,
        skew_threshold_mild: float = 0.5,
        skew_threshold_high: float = 2.0,
    ) -> DataFrame:
        """Apply sampling to DataFrame if dataset exceeds threshold.

        Supports adaptive sampling (v2.9.0) which selects strategy based on
        data skewness to better preserve tail information for skewed distributions.

        Args:
            df: DataFrame to sample (Spark or pandas)
            row_count: Total row count of DataFrame
            enable_sampling: Whether sampling is enabled
            sample_fraction: Explicit fraction to sample (None = auto-determine)
            max_sample_size: Maximum rows when auto-determining fraction
            sample_threshold: Row count above which sampling is applied
            column: Column to analyze for adaptive sampling (required if adaptive)
            adaptive_sampling: Whether to use adaptive sampling based on skewness
            sampling_mode: Sampling strategy (AUTO, UNIFORM, or STRATIFIED)
            skew_threshold_mild: Skewness threshold for mild stratification
            skew_threshold_high: Skewness threshold for aggressive stratification

        Returns:
            Original DataFrame if no sampling needed, otherwise sampled DataFrame
        """
        if not enable_sampling or row_count <= sample_threshold:
            return df

        if sample_fraction is not None:
            fraction = sample_fraction
        else:
            fraction = min(max_sample_size / row_count, 0.35)

        # Determine effective sampling mode
        effective_mode = sampling_mode
        skewness = None
        if adaptive_sampling and sampling_mode == SamplingMode.AUTO and column is not None:
            skewness = self._compute_skewness_estimate(df, column, row_count)
            effective_mode = self._select_sampling_strategy(skewness, skew_threshold_mild, skew_threshold_high)
            logger.info(f"Adaptive sampling: skewness={skewness:.2f}, mode={effective_mode.value}")

        # Apply appropriate sampling strategy
        if effective_mode == SamplingMode.STRATIFIED and column is not None:
            # Determine number of bins based on skewness (reuse computed value if available)
            if skewness is None:
                skewness = self._compute_skewness_estimate(df, column, row_count)
            abs_skew = abs(skewness)
            n_bins = 10 if abs_skew >= skew_threshold_high else 5
            return self._apply_stratified_sampling(df, column, fraction, row_count, n_bins)

        # Default: uniform sampling
        logger.info(f"Sampling {fraction * 100:.1f}% of data ({int(row_count * fraction)} rows)")
        if hasattr(df, "sparkSession"):
            return df.sample(fraction=fraction, seed=self.random_seed)
        else:
            return df.sample(frac=fraction, random_state=self.random_seed)

    # Minimum pilot sample size for stable skewness estimate
    _PILOT_SAMPLE_SIZE: int = 5000

    def _compute_skewness_estimate(
        self,
        df: DataFrame,
        column: str,
        row_count: int,
    ) -> float:
        """Compute skewness estimate from a pilot sample.

        Uses a small pilot sample (minimum 5000 points) to estimate data skewness
        efficiently without a full data pass. For datasets smaller than pilot size,
        computes exact skewness.

        Args:
            df: DataFrame containing the data
            column: Column to analyze
            row_count: Total row count

        Returns:
            Estimated skewness value
        """
        # Determine pilot sample fraction
        if row_count <= self._PILOT_SAMPLE_SIZE:
            pilot_fraction = 1.0
        else:
            pilot_fraction = self._PILOT_SAMPLE_SIZE / row_count

        # Extract pilot sample
        if hasattr(df, "sparkSession"):
            # Spark DataFrame
            if pilot_fraction < 1.0:
                pilot_df = df.sample(fraction=pilot_fraction, seed=self.random_seed)
            else:
                pilot_df = df
            pilot_data = np.array([row[column] for row in pilot_df.select(column).collect()])
        else:
            # pandas DataFrame
            if pilot_fraction < 1.0:
                pilot_df = df.sample(frac=pilot_fraction, random_state=self.random_seed)
            else:
                pilot_df = df
            pilot_data = pilot_df[column].values

        # Compute skewness
        skewness = float(stats.skew(pilot_data, nan_policy="omit"))

        # Log warning for extreme skew (Mayor feedback)
        if abs(skewness) > 5.0:
            logger.warning(
                f"Extreme skew detected (|s|={abs(skewness):.1f} > 5): " "aggressive tail boosting will be activated"
            )

        return skewness

    def _select_sampling_strategy(
        self,
        skewness: float,
        skew_threshold_mild: float,
        skew_threshold_high: float,
    ) -> SamplingMode:
        """Select sampling strategy based on data skewness.

        Args:
            skewness: Estimated data skewness
            skew_threshold_mild: Threshold for mild stratification (default 0.5)
            skew_threshold_high: Threshold for aggressive stratification (default 2.0)

        Returns:
            SamplingMode.UNIFORM for symmetric data, SamplingMode.STRATIFIED for skewed
        """
        abs_skew = abs(skewness)
        if abs_skew < skew_threshold_mild:
            return SamplingMode.UNIFORM
        else:
            return SamplingMode.STRATIFIED

    def _apply_stratified_sampling(
        self,
        df: DataFrame,
        column: str,
        fraction: float,
        row_count: int,
        n_bins: int = 10,
    ) -> DataFrame:
        """Apply stratified sampling across percentile bins.

        Divides data into bins based on percentiles and samples proportionally
        from each bin to preserve the distribution shape, especially tail regions.

        Args:
            df: DataFrame to sample
            column: Column to stratify on
            fraction: Overall sample fraction
            row_count: Total row count
            n_bins: Number of percentile bins (5 for mild skew, 10 for high skew)

        Returns:
            Stratified sampled DataFrame
        """
        # Calculate target sample size
        target_size = int(row_count * fraction)

        # Minimum samples per tail bin (first and last) - Mayor feedback
        min_tail_samples = 500

        if hasattr(df, "sparkSession"):
            return self._apply_stratified_sampling_spark(df, column, fraction, n_bins, target_size, min_tail_samples)
        else:
            return self._apply_stratified_sampling_pandas(df, column, fraction, n_bins, target_size, min_tail_samples)

    def _apply_stratified_sampling_spark(
        self,
        df: DataFrame,
        column: str,
        fraction: float,
        n_bins: int,
        target_size: int,
        min_tail_samples: int,
    ) -> DataFrame:
        """Apply stratified sampling for Spark DataFrames."""
        from pyspark.ml.feature import Bucketizer

        # Compute approximate percentiles
        percentiles = [i / n_bins for i in range(n_bins + 1)]
        splits = df.approxQuantile(column, percentiles, 0.01)

        # Ensure splits are strictly increasing (handle edge cases)
        splits = sorted(set(splits))
        if len(splits) < 2:
            # Fallback to uniform if quantiles collapse
            logger.warning("Quantiles collapsed, falling back to uniform sampling")
            return df.sample(fraction=fraction, seed=self.random_seed)

        # Add -inf and +inf for bucketizer
        splits = [float("-inf")] + splits[1:-1] + [float("inf")]

        # Bucketize the data
        bucketizer = Bucketizer(splits=splits, inputCol=column, outputCol="_strat_bucket")
        bucketed_df = bucketizer.transform(df)

        # Sample from each bucket
        # Use sampleBy for stratified sampling
        fractions = {float(i): fraction for i in range(len(splits) - 1)}

        # Boost tail fractions (first and last bucket)
        # Ensure minimum tail samples
        bucket_counts = bucketed_df.groupBy("_strat_bucket").count().collect()
        bucket_count_map = {int(row["_strat_bucket"]): row["count"] for row in bucket_counts}

        for tail_bucket in [0, len(splits) - 2]:
            if tail_bucket in bucket_count_map:
                tail_count = bucket_count_map[tail_bucket]
                if tail_count > 0:
                    min_fraction = min(min_tail_samples / tail_count, 1.0)
                    fractions[float(tail_bucket)] = max(fractions[float(tail_bucket)], min_fraction)

        sampled = bucketed_df.sampleBy("_strat_bucket", fractions, seed=self.random_seed)
        sampled = sampled.drop("_strat_bucket")

        actual_size = sampled.count()
        logger.info(
            f"Stratified sampling: {n_bins} bins, "
            f"{actual_size} rows ({actual_size * 100 / target_size:.1f}% of target)"
        )

        return sampled

    def _apply_stratified_sampling_pandas(
        self,
        df: DataFrame,
        column: str,
        fraction: float,
        n_bins: int,
        target_size: int,
        min_tail_samples: int,
    ) -> DataFrame:
        """Apply stratified sampling for pandas DataFrames."""
        import pandas as pd

        # Create percentile bins
        df = df.copy()
        df["_strat_bucket"] = pd.qcut(df[column], q=n_bins, labels=False, duplicates="drop")

        # Sample from each bucket
        sampled_parts = []
        unique_buckets = df["_strat_bucket"].dropna().unique()

        for bucket in unique_buckets:
            bucket_df = df[df["_strat_bucket"] == bucket]
            bucket_size = len(bucket_df)

            # Calculate sample size for this bucket
            sample_size = int(bucket_size * fraction)

            # Boost tail buckets
            if bucket == unique_buckets.min() or bucket == unique_buckets.max():
                sample_size = max(sample_size, min(min_tail_samples, bucket_size))

            if sample_size > 0:
                sampled = bucket_df.sample(n=min(sample_size, bucket_size), random_state=self.random_seed)
                sampled_parts.append(sampled)

        result = pd.concat(sampled_parts, ignore_index=True)
        result = result.drop(columns=["_strat_bucket"])

        logger.info(
            f"Stratified sampling: {n_bins} bins, "
            f"{len(result)} rows ({len(result) * 100 / target_size:.1f}% of target)"
        )

        return result

    def _calculate_partitions(self, distributions: List[str]) -> int:
        """Calculate optimal partition count for distribution fitting.

        Uses distribution-aware weighting: slow distributions count as 3x
        for partition calculation to reduce straggler effects when slow
        distributions cluster in the same partition.

        Args:
            distributions: List of distribution names to fit

        Returns:
            Optimal partition count for the fitting operation
        """
        from spark_bestfit.distributions import DistributionRegistry

        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        slow_count = sum(1 for d in distributions if d in slow_set)
        # Slow distributions count as 3x (add 2 extra for each slow one)
        effective_count = len(distributions) + slow_count * 2
        total_cores = self._backend.get_parallelism()
        return min(effective_count, total_cores * 2)

    @staticmethod
    def _validate_column_exists(
        df: DataFrame,
        column: str,
    ) -> None:
        """Validate that a column exists in the DataFrame.

        Args:
            df: DataFrame to check
            column: Column name to validate

        Raises:
            ValueError: If column not found in DataFrame
        """
        # Get columns list - handle Spark, pandas, and Ray Dataset
        if hasattr(df, "select_columns") and hasattr(df, "schema"):
            # Ray Dataset - use schema() method to get column names
            columns_list = df.schema().names
        elif hasattr(df, "sparkSession"):
            # Spark DataFrame
            columns_list = df.columns
        else:
            # pandas DataFrame (or similar)
            columns_list = list(df.columns)

        if column not in columns_list:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {columns_list}")

    @staticmethod
    def _validate_column_numeric(
        df: DataFrame,
        column: str,
    ) -> None:
        """Validate that a column is numeric.

        Args:
            df: DataFrame to check
            column: Column name to validate

        Raises:
            TypeError: If column is not numeric
        """
        # Handle type checking for Spark, pandas, and Ray Dataset
        if hasattr(df, "select_columns") and hasattr(df, "schema"):
            # Ray Dataset - check schema for numeric type
            schema = df.schema()
            col_idx = schema.names.index(column)
            col_type = schema.types[col_idx]
            # Ray DataType has string repr like 'double', 'int64', etc.
            type_str = str(col_type).lower()
            numeric_types = ("int", "float", "double", "decimal")
            if not any(t in type_str for t in numeric_types):
                raise TypeError(f"Column '{column}' must be numeric, got {col_type}")
        elif hasattr(df, "sparkSession") and _PYSPARK_AVAILABLE:
            # Spark DataFrame
            col_type = df.schema[column].dataType
            if not isinstance(col_type, NumericType):
                raise TypeError(f"Column '{column}' must be numeric, got {col_type}")
        else:
            # pandas DataFrame (or similar)
            import pandas as pd

            if hasattr(df, "dtypes"):
                col_dtype = df[column].dtype
                if not pd.api.types.is_numeric_dtype(col_dtype):
                    raise TypeError(f"Column '{column}' must be numeric, got {col_dtype}")

    @staticmethod
    def _validate_sample_fraction(sample_fraction: Optional[float]) -> None:
        """Validate sample_fraction parameter.

        Args:
            sample_fraction: Sampling fraction to validate

        Raises:
            ValueError: If sample_fraction is out of range (0, 1]
        """
        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    @staticmethod
    def _validate_max_distributions(max_distributions: Optional[int]) -> None:
        """Validate max_distributions parameter.

        Args:
            max_distributions: Maximum distributions to validate

        Raises:
            ValueError: If max_distributions is 0
        """
        if max_distributions == 0:
            raise ValueError("max_distributions cannot be 0")

    @staticmethod
    def _get_row_count(df: DataFrame) -> int:
        """Get row count from DataFrame, handling multiple DataFrame types.

        Args:
            df: DataFrame (Spark, pandas, or Ray Dataset)

        Returns:
            Row count of the DataFrame
        """
        if hasattr(df, "sparkSession"):
            return df.count()
        elif hasattr(df, "select_columns") and hasattr(df, "count"):
            # Ray Dataset - use count() method
            return df.count()
        else:
            return len(df)

    def _normalize_columns(
        self,
        column: Optional[str],
        columns: Optional[List[str]],
    ) -> List[str]:
        """Normalize column/columns parameters to a list.

        Args:
            column: Single column name (mutually exclusive with columns)
            columns: List of column names (mutually exclusive with column)

        Returns:
            List of column names

        Raises:
            ValueError: If neither or both parameters are provided
        """
        if column is None and columns is None:
            raise ValueError("Must provide either 'column' or 'columns' parameter")
        if column is not None and columns is not None:
            raise ValueError("Cannot provide both 'column' and 'columns' - use one or the other")

        return [column] if column is not None else columns
