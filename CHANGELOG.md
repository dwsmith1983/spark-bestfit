# CHANGELOG

<!-- version list -->

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
