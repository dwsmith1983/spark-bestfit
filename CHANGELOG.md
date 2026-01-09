# CHANGELOG

<!-- version list -->

## v2.4.0 (2026-01-09)

### Features

- Add custom distribution support (#15)
  ([#102](https://github.com/dwsmith1983/spark-bestfit/pull/102),
  [`9cee327`](https://github.com/dwsmith1983/spark-bestfit/commit/9cee327a5c1827aa81a639c36364df202eb1bfaa))


## v2.3.0 (2026-01-08)

### Bug Fixes

- **ci**: Exclude Spark-specific code branches from coverage
  ([#98](https://github.com/dwsmith1983/spark-bestfit/pull/98),
  [`a590c58`](https://github.com/dwsmith1983/spark-bestfit/commit/a590c58f664475c9c7d9f73c409de754468d571d))

- **ci**: Exclude Spark/Ray backends from CI coverage threshold
  ([#98](https://github.com/dwsmith1983/spark-bestfit/pull/98),
  [`a590c58`](https://github.com/dwsmith1983/spark-bestfit/commit/a590c58f664475c9c7d9f73c409de754468d571d))

- **ci**: Expand coverage exclusions for Spark-only modules
  ([#98](https://github.com/dwsmith1983/spark-bestfit/pull/98),
  [`a590c58`](https://github.com/dwsmith1983/spark-bestfit/commit/a590c58f664475c9c7d9f73c409de754468d571d))

### Features

- Add heavy-tail distribution detection and warnings
  ([#98](https://github.com/dwsmith1983/spark-bestfit/pull/98),
  [`a590c58`](https://github.com/dwsmith1983/spark-bestfit/commit/a590c58f664475c9c7d9f73c409de754468d571d))

- V2.3.0 DX Foundation - examples, tests, heavy-tail detection, docs
  ([#98](https://github.com/dwsmith1983/spark-bestfit/pull/98),
  [`a590c58`](https://github.com/dwsmith1983/spark-bestfit/commit/a590c58f664475c9c7d9f73c409de754468d571d))


## v2.2.0 (2026-01-08)

### Bug Fixes

- Add proper skip markers for optional dependency tests
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- Exclude hypothesis tests from mutmut ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- Improve hypothesis test stability for mutmut
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- Make ray import conditional in test_backend_factory.py
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- Pin mutmut <3.0 to avoid trampoline segfaults
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

### Features

- Add BackendFactory and deprecate sample_spark() methods
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- Add FitterConfig builder pattern for cleaner API
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- V2.2.0 - FitterConfig, BackendFactory, TruncatedFrozenDist consolidation
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

### Refactoring

- Consolidate TruncatedFrozenDist implementations into single module
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))

- Remove orphaned code and consolidate duplicates
  ([#93](https://github.com/dwsmith1983/spark-bestfit/pull/93),
  [`5da0a95`](https://github.com/dwsmith1983/spark-bestfit/commit/5da0a95e5817568a74550313a292c23df2d86869))


## v2.1.0 (2026-01-07)

### Bug Fixes

- Adjust shape parameter minimums for numerical stability in property tests
  ([#92](https://github.com/dwsmith1983/spark-bestfit/pull/92),
  [`45c1f91`](https://github.com/dwsmith1983/spark-bestfit/commit/45c1f911dc8fc3035e71ac191f674d021d4f9aaa))

### Features

- Split FitResults into lazy/eager hierarchy + property-based testing
  ([#92](https://github.com/dwsmith1983/spark-bestfit/pull/92),
  [`45c1f91`](https://github.com/dwsmith1983/spark-bestfit/commit/45c1f911dc8fc3035e71ac191f674d021d4f9aaa))

### Refactoring

- Add property-based testing with hypothesis
  ([#92](https://github.com/dwsmith1983/spark-bestfit/pull/92),
  [`45c1f91`](https://github.com/dwsmith1983/spark-bestfit/commit/45c1f911dc8fc3035e71ac191f674d021d4f9aaa))

- Split FitResults into lazy/eager class hierarchy
  ([#92](https://github.com/dwsmith1983/spark-bestfit/pull/92),
  [`45c1f91`](https://github.com/dwsmith1983/spark-bestfit/commit/45c1f911dc8fc3035e71ac191f674d021d4f9aaa))


## v2.0.3 (2026-01-06)

### Bug Fixes

- CI fail-fast and Spark test imports ([#91](https://github.com/dwsmith1983/spark-bestfit/pull/91),
  [`19dc28c`](https://github.com/dwsmith1983/spark-bestfit/commit/19dc28cc465051b9856b801039a3d2f9023b8625))

- Correct mutmut test selection to exclude Spark-dependent tests
  ([#91](https://github.com/dwsmith1983/spark-bestfit/pull/91),
  [`19dc28c`](https://github.com/dwsmith1983/spark-bestfit/commit/19dc28cc465051b9856b801039a3d2f9023b8625))

### Refactoring

- LocalBackend tests and backend auto-detection for mutation testing
  ([#91](https://github.com/dwsmith1983/spark-bestfit/pull/91),
  [`19dc28c`](https://github.com/dwsmith1983/spark-bestfit/commit/19dc28cc465051b9856b801039a3d2f9023b8625))

- Separate Spark tests from LocalBackend tests for mutation testing
  ([#91](https://github.com/dwsmith1983/spark-bestfit/pull/91),
  [`19dc28c`](https://github.com/dwsmith1983/spark-bestfit/commit/19dc28cc465051b9856b801039a3d2f9023b8625))


## v2.0.2 (2026-01-05)

### Bug Fixes

- Add pyarrow back for Ray tests (required by ray.data)
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Add pyarrow to ray optional dependency (required by ray.data)
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Handle NaN/inf values and empty results in distribution fitting
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Make PySpark optional in continuous_fitter, discrete_fitter, utils
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Make PySpark optional in progress.py ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Make SparkBackend import conditional for Ray-only environments
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Remove pyarrow from Ray test matrix (not needed)
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Use future annotations to defer Broadcast type evaluation
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

### Refactoring

- BaseFitter extraction, numerical stability, LazyMetrics lifecycle
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Extract BaseFitter base class to eliminate code duplication
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))

- Make PySpark truly optional for Ray/Local backends
  ([#89](https://github.com/dwsmith1983/spark-bestfit/pull/89),
  [`e3124f3`](https://github.com/dwsmith1983/spark-bestfit/commit/e3124f3618743abb0e66246716cdcdd2034d269a))


## v2.0.1 (2026-01-05)

### Bug Fixes

- Add matplotlib to test dependencies for CI
  ([`82fad83`](https://github.com/dwsmith1983/spark-bestfit/commit/82fad838ffc0ccdebd5528a0e062d0017148e932))

- Add matplotlib to test-base for CI
  ([`82fad83`](https://github.com/dwsmith1983/spark-bestfit/commit/82fad838ffc0ccdebd5528a0e062d0017148e932))


## v2.0.0 (2026-01-04)

### Bug Fixes

- Improve backend parity, exception handling, and thread safety
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

- **docs**: Update usecases.rst references to features/ subdirectory
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

### Features

- Abstract sampling module to use backend protocol
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

- Add ExecutionBackend protocol for multi-backend support
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

- Add RayBackend for distributed fitting on Ray clusters
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

- Add unified progress_callback support across all backends
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

- Extend backend protocol for copula and histogram support
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

- V2.0.0 - Multi-backend architecture with Ray support
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

### Performance Improvements

- **schema**: Flatten data_summary MapType to individual columns
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

### Refactoring

- Split core.py into separate fitter modules
  ([#86](https://github.com/dwsmith1983/spark-bestfit/pull/86),
  [`282df3b`](https://github.com/dwsmith1983/spark-bestfit/commit/282df3b905851612cecf7562e421c9eee60e1079))

### Breaking Changes

- **schema**: Data_summary dict replaced with individual attributes


## v1.7.2 (2026-01-02)

### Bug Fixes

- Correct exponpow typo and modernize GitHub templates
  ([#81](https://github.com/dwsmith1983/spark-bestfit/pull/81),
  [`d4697f2`](https://github.com/dwsmith1983/spark-bestfit/commit/d4697f2de97057fdc4726a08194735d933c4e2cf))


## v1.7.1 (2026-01-02)

### Bug Fixes

- Excluded_distributions parameter now properly overrides registry defaults
  ([#79](https://github.com/dwsmith1983/spark-bestfit/pull/79),
  [`320a615`](https://github.com/dwsmith1983/spark-bestfit/commit/320a61584c0b92c6639ca260719c1bc5289f1919))


## v1.7.0 (2026-01-02)

### Features

- **perf**: Add distribution-aware partitioning for better load balancing
  ([#76](https://github.com/dwsmith1983/spark-bestfit/pull/76),
  [`c42fc24`](https://github.com/dwsmith1983/spark-bestfit/commit/c42fc24f566cb8832487db24de547ad2116c50ce))


## v1.6.0 (2026-01-01)

### Features

- Add prefilter parameter and CDF-based PDF integration
  ([#75](https://github.com/dwsmith1983/spark-bestfit/pull/75),
  [`c418d47`](https://github.com/dwsmith1983/spark-bestfit/commit/c418d47c733a703c4f23352ab65577f9c26bf0f6))

- Add prefilter parameter for shape-based distribution filtering
  ([#75](https://github.com/dwsmith1983/spark-bestfit/pull/75),
  [`c418d47`](https://github.com/dwsmith1983/spark-bestfit/commit/c418d47c733a703c4f23352ab65577f9c26bf0f6))

### Performance Improvements

- CDF-based PDF integration and memory management improvements
  ([#75](https://github.com/dwsmith1983/spark-bestfit/pull/75),
  [`c418d47`](https://github.com/dwsmith1983/spark-bestfit/commit/c418d47c733a703c4f23352ab65577f9c26bf0f6))


## v1.5.1 (2026-01-01)

### Bug Fixes

- Trigger v1.5.1 release for cache warning fix
  ([#74](https://github.com/dwsmith1983/spark-bestfit/pull/74),
  [`45b6101`](https://github.com/dwsmith1983/spark-bestfit/commit/45b61011a6f745ccd95e3374884f9c7d01eb6103))


## v1.5.0 (2026-01-01)

### Features

- Add lazy metrics for on-demand KS/AD computation
  ([#72](https://github.com/dwsmith1983/spark-bestfit/pull/72),
  [`fecc18d`](https://github.com/dwsmith1983/spark-bestfit/commit/fecc18dd5775f98c09f17de5b011ce176371f07f))

- Add per-column bounds for multi-column fitting
  ([#72](https://github.com/dwsmith1983/spark-bestfit/pull/72),
  [`fecc18d`](https://github.com/dwsmith1983/spark-bestfit/commit/fecc18dd5775f98c09f17de5b011ce176371f07f))

- V1.5.0 - lazy metrics and per-column bounds
  ([#72](https://github.com/dwsmith1983/spark-bestfit/pull/72),
  [`fecc18d`](https://github.com/dwsmith1983/spark-bestfit/commit/fecc18dd5775f98c09f17de5b011ce176371f07f))


## v1.4.0 (2025-12-31)

### Features

- Add bounded distribution fitting for continuous distributions
  ([#61](https://github.com/dwsmith1983/spark-bestfit/pull/61),
  [`dd129d5`](https://github.com/dwsmith1983/spark-bestfit/commit/dd129d5b2ec98f4dc95199e5b21837ca907f64be))

- Add bounded distribution fitting for DiscreteDistributionFitter
  ([#61](https://github.com/dwsmith1983/spark-bestfit/pull/61),
  [`dd129d5`](https://github.com/dwsmith1983/spark-bestfit/commit/dd129d5b2ec98f4dc95199e5b21837ca907f64be))

- Add bounded/truncated distribution fitting (v1.4.0)
  ([#61](https://github.com/dwsmith1983/spark-bestfit/pull/61),
  [`dd129d5`](https://github.com/dwsmith1983/spark-bestfit/commit/dd129d5b2ec98f4dc95199e5b21837ca907f64be))


## v1.3.2 (2025-12-31)

### Performance Improvements

- V1.3.2 performance optimizations (#53, #54, #57)
  ([#58](https://github.com/dwsmith1983/spark-bestfit/pull/58),
  [`5f3ec48`](https://github.com/dwsmith1983/spark-bestfit/commit/5f3ec48ef793a3af184f598fe20189a0924c8058))


## v1.3.1 (2025-12-31)

### Bug Fixes

- **sampling**: Replace iterrows() with iloc for performance
  ([#52](https://github.com/dwsmith1983/spark-bestfit/pull/52),
  [`c51f784`](https://github.com/dwsmith1983/spark-bestfit/commit/c51f784a4ef34536a020e573845eb75cd170e0ed))


## v1.3.0 (2025-12-30)

### Features

- Add distributed sampling and fit quality warnings (#43, #45)
  ([#50](https://github.com/dwsmith1983/spark-bestfit/pull/50),
  [`f696d6a`](https://github.com/dwsmith1983/spark-bestfit/commit/f696d6a767e05863711a137d4fd2ab886639d955))

- Add Gaussian Copula for correlated multi-column sampling
  ([#50](https://github.com/dwsmith1983/spark-bestfit/pull/50),
  [`f696d6a`](https://github.com/dwsmith1983/spark-bestfit/commit/f696d6a767e05863711a137d4fd2ab886639d955))

- Add serialization support for fitted distributions
  ([#50](https://github.com/dwsmith1983/spark-bestfit/pull/50),
  [`f696d6a`](https://github.com/dwsmith1983/spark-bestfit/commit/f696d6a767e05863711a137d4fd2ab886639d955))

- V1.3.0 Downstream Enablement ([#50](https://github.com/dwsmith1983/spark-bestfit/pull/50),
  [`f696d6a`](https://github.com/dwsmith1983/spark-bestfit/commit/f696d6a767e05863711a137d4fd2ab886639d955))

### Performance Improvements

- **copula**: Optimize sampling with frozen distributions and return_uniform
  ([#50](https://github.com/dwsmith1983/spark-bestfit/pull/50),
  [`f696d6a`](https://github.com/dwsmith1983/spark-bestfit/commit/f696d6a767e05863711a137d4fd2ab886639d955))


## v1.2.0 (2025-12-30)

### Features

- Add progress tracking for distribution fitting (#12)
  ([#41](https://github.com/dwsmith1983/spark-bestfit/pull/41),
  [`5477558`](https://github.com/dwsmith1983/spark-bestfit/commit/5477558d718f9cdb48243b64b68b4f4e3b84ebbb))


## v1.1.0 (2025-12-29)

### Features

- Multi-column fitting in single call (#10)
  ([#38](https://github.com/dwsmith1983/spark-bestfit/pull/38),
  [`c58ebc6`](https://github.com/dwsmith1983/spark-bestfit/commit/c58ebc60ad27f6a658b4c1cc1518784377007ec2))

- **core**: Add multi-column fitting support
  ([#38](https://github.com/dwsmith1983/spark-bestfit/pull/38),
  [`c58ebc6`](https://github.com/dwsmith1983/spark-bestfit/commit/c58ebc60ad27f6a658b4c1cc1518784377007ec2))

- **docs**: Add multi-column fitting documentation and benchmarks
  ([#38](https://github.com/dwsmith1983/spark-bestfit/pull/38),
  [`c58ebc6`](https://github.com/dwsmith1983/spark-bestfit/commit/c58ebc60ad27f6a658b4c1cc1518784377007ec2))

- **results**: Add column_name tracking for multi-column fitting
  ([#38](https://github.com/dwsmith1983/spark-bestfit/pull/38),
  [`c58ebc6`](https://github.com/dwsmith1983/spark-bestfit/commit/c58ebc60ad27f6a658b4c1cc1518784377007ec2))


## v1.0.0 (2025-12-28)

### Features

- 1.0.0 release preparation ([#37](https://github.com/dwsmith1983/spark-bestfit/pull/37),
  [`47e662f`](https://github.com/dwsmith1983/spark-bestfit/commit/47e662f8e0bbeeb779e2e29073d159780184191e))


## v0.8.0 (2025-12-26)

### Features

- Add performance benchmarks and scaling documentation
  ([#33](https://github.com/dwsmith1983/spark-bestfit/pull/33),
  [`89dc763`](https://github.com/dwsmith1983/spark-bestfit/commit/89dc76337e30b4a980601db012b8d7898d1824cd))


## v0.7.2 (2025-12-26)

### Bug Fixes

- Broadcast cleanup bug and add failure mode tests
  ([#31](https://github.com/dwsmith1983/spark-bestfit/pull/31),
  [`b5b2a53`](https://github.com/dwsmith1983/spark-bestfit/commit/b5b2a538653bea6a51daacc1d8623c71527f9a79))


## v0.7.1 (2025-12-25)

### Bug Fixes

- Add IQR-based outlier filtering to bootstrap confidence intervals
  ([#30](https://github.com/dwsmith1983/spark-bestfit/pull/30),
  [`6f1d582`](https://github.com/dwsmith1983/spark-bestfit/commit/6f1d58276c38cdc4ee8bdf9206ea5ccb78180ea4))


## v0.7.0 (2025-12-25)

### Features

- Add bootstrap confidence intervals for fitted parameters (#11)
  ([#29](https://github.com/dwsmith1983/spark-bestfit/pull/29),
  [`a7ff425`](https://github.com/dwsmith1983/spark-bestfit/commit/a7ff425cc7f384272f2c8ad266a9e8a98cfc60ea))


## v0.6.0 (2025-12-25)

### Features

- Add Anderson-Darling goodness-of-fit test
  ([#27](https://github.com/dwsmith1983/spark-bestfit/pull/27),
  [`ba63f25`](https://github.com/dwsmith1983/spark-bestfit/commit/ba63f251e0239513a29d144cdff124e1371017d6))


## v0.5.0 (2025-12-25)

### Bug Fixes

- Address p-value consistency and resolve type errors
  ([#23](https://github.com/dwsmith1983/spark-bestfit/pull/23),
  [`6df35a6`](https://github.com/dwsmith1983/spark-bestfit/commit/6df35a65d20c52d9179efccb509852fdad1cc4ff))

- Finalize distribution registry integration and clean up core.py
  ([#23](https://github.com/dwsmith1983/spark-bestfit/pull/23),
  [`6df35a6`](https://github.com/dwsmith1983/spark-bestfit/commit/6df35a65d20c52d9179efccb509852fdad1cc4ff))

### Features

- Add P-P plots for goodness-of-fit assessment
  ([#23](https://github.com/dwsmith1983/spark-bestfit/pull/23),
  [`6df35a6`](https://github.com/dwsmith1983/spark-bestfit/commit/6df35a65d20c52d9179efccb509852fdad1cc4ff))

- Add P-P plots for goodness-of-fit assessment (#9)
  ([#23](https://github.com/dwsmith1983/spark-bestfit/pull/23),
  [`6df35a6`](https://github.com/dwsmith1983/spark-bestfit/commit/6df35a65d20c52d9179efccb509852fdad1cc4ff))


## v0.4.0 (2025-12-24)

### Features

- Add discrete distribution fitting with 16 distributions
  ([#22](https://github.com/dwsmith1983/spark-bestfit/pull/22),
  [`56693f8`](https://github.com/dwsmith1983/spark-bestfit/commit/56693f83c1e202c98c9dd3cdcea72165eba7c367))


## v0.3.1 (2025-12-17)

### Performance Improvements

- Optimize Spark actions in plot_qq and filter
  ([#19](https://github.com/dwsmith1983/spark-bestfit/pull/19),
  [`26ffa66`](https://github.com/dwsmith1983/spark-bestfit/commit/26ffa663732c9150182b688660ba67c959f78e7b))


## v0.3.0 (2025-12-17)

### Features

- Add Q-Q plots for goodness-of-fit assessment
  ([#7](https://github.com/dwsmith1983/spark-bestfit/pull/7),
  [`ca23d8d`](https://github.com/dwsmith1983/spark-bestfit/commit/ca23d8d85f3c2ed66854b1da1134915d2f4d4d8d))


## v0.2.1 (2025-12-17)

### Bug Fixes

- Unnecessary, superflouos pandas wrapper
  ([#6](https://github.com/dwsmith1983/spark-bestfit/pull/6),
  [`9dfbc6c`](https://github.com/dwsmith1983/spark-bestfit/commit/9dfbc6c656504796a9a653e24cb78f4d2eacdbc7))


## v0.2.0 (2025-12-16)

### Features

- Add Kolmogorov-Smirnov statistic and p-values for goodness-of-fit
  ([#5](https://github.com/dwsmith1983/spark-bestfit/pull/5),
  [`974a3a3`](https://github.com/dwsmith1983/spark-bestfit/commit/974a3a3a6b486a55228dde10175696d34684216b))


## v0.1.1 (2025-12-14)

### Bug Fixes

- Trigger release for PyPI README update ([#4](https://github.com/dwsmith1983/spark-bestfit/pull/4),
  [`1aeac86`](https://github.com/dwsmith1983/spark-bestfit/commit/1aeac869904a1e1cf9515d681227d50414bccfdc))


## v0.1.0 (2025-12-14)

- Initial Release
