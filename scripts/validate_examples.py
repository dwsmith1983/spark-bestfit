#!/usr/bin/env python
"""Validate that example scripts have valid imports and syntax.

This script checks:
1. Syntax is valid (AST parses)
2. All imports from spark_bestfit resolve to actual exports

Note: This uses AST parsing instead of importing to avoid requiring pyspark.
"""

import ast
import sys
from pathlib import Path

# Root of the package
PACKAGE_ROOT = Path(__file__).parent.parent / "src" / "spark_bestfit"


def get_package_exports() -> set[str]:
    """Get all public exports from spark_bestfit/__init__.py using AST.

    Parses the __init__.py to find:
    - Names in __all__ if defined
    - Otherwise, all imported names from 'from x import y' statements
    """
    init_file = PACKAGE_ROOT / "__init__.py"

    with open(init_file) as f:
        source = f.read()

    tree = ast.parse(source)
    exports = set()

    # First, look for __all__
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                exports.add(elt.value)

    # If __all__ found, use it
    if exports:
        return exports

    # Otherwise, collect all imported names
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if not name.startswith("_"):
                    exports.add(name)

    return exports


def get_spark_bestfit_imports(filepath: Path) -> list[tuple[str, int]]:
    """Extract all imports from spark_bestfit in a file.

    Returns list of (import_name, line_number) tuples.
    """
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"FAIL: {filepath} - Syntax error: {e}")
        sys.exit(1)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("spark_bestfit"):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spark_bestfit"):
                    imports.append((alias.name, node.lineno))

    return imports


def validate_imports(imports: list[tuple[str, int]], filepath: Path, public_exports: set[str]) -> bool:
    """Validate that all imports exist in spark_bestfit."""
    valid = True
    for name, lineno in imports:
        if name not in public_exports:
            print(f"FAIL: {filepath}:{lineno} - '{name}' not found in spark_bestfit")
            print(f"      Available exports: {sorted(public_exports)}")
            valid = False

    return valid


def main():
    examples_dir = Path(__file__).parent.parent / "examples"
    py_files = list(examples_dir.glob("*.py"))

    if not py_files:
        print(f"No Python files found in {examples_dir}")
        sys.exit(1)

    # Get package exports once (using AST, no pyspark needed)
    public_exports = get_package_exports()
    print(f"Package exports: {sorted(public_exports)}")
    print(f"\nValidating {len(py_files)} example files...")

    all_valid = True

    for filepath in py_files:
        print(f"\nChecking {filepath.name}...")

        # Get imports
        imports = get_spark_bestfit_imports(filepath)

        if not imports:
            print("  No spark_bestfit imports found")
            continue

        print(f"  Found imports: {[name for name, _ in imports]}")

        # Validate imports exist
        if not validate_imports(imports, filepath, public_exports):
            all_valid = False
        else:
            print("  OK - all imports valid")

    print("\n" + "=" * 60)
    if all_valid:
        print("SUCCESS: All example imports are valid")
        sys.exit(0)
    else:
        print("FAILURE: Some imports are invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
