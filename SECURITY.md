# Security Policy

## Dependency Vulnerability Scanning

This project uses [pip-audit](https://github.com/pypa/pip-audit) to scan for known CVEs in dependencies. The scan runs on every pull request.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Known Transitive CVEs (From PySpark/PyArrow)

The following CVEs may appear in **transitive dependencies from PySpark and PyArrow**. These are inherited from the Spark ecosystem and cannot be resolved by this project directly.

### PySpark (Transitive)

PySpark bundles Java dependencies (Spark, Hadoop, Netty) that may have known vulnerabilities. These are managed by the Apache Spark project.

**Mitigation**: Keep PySpark updated to the latest patch version. Ensure your Spark cluster network is secured.

### PyArrow (Transitive)

PyArrow is required for Pandas UDFs in Spark. CVEs in Arrow's C++ core are typically patched quickly.

**Mitigation**: Keep PyArrow updated. The library is used only for data serialization between Python and Spark.

## Security Considerations

spark-bestfit processes numerical data for statistical analysis. Key considerations:

- **Data Privacy**: All computation happens within your Spark cluster; no data is sent externally
- **Dependencies**: We use scipy, numpy, pandas, and matplotlib - all well-audited scientific Python libraries
- **No Network Access**: The library makes no network calls during fitting operations
- **No Code Execution**: The library does not execute arbitrary code; only scipy.stats distributions are fitted

## Reporting Security Issues

If you discover a security vulnerability in this project (not transitive dependencies), please report it by:

1. **GitHub**: Open a private security advisory on GitHub
2. **Do NOT** open a public issue for security vulnerabilities

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

For vulnerabilities in PySpark, PyArrow, or other upstream projects, please report to the respective Apache/PyPI security teams.

## Updates

This document was last updated: 2025-12-28
