# Common utilities for ISDA-SIMM Benchmark
# Generic functions reusable across different model implementations
"""
Utility module for ISDA-SIMM Benchmark.

Provides generic utilities for code analysis and file operations
that can be reused across different model implementations.
"""

from typing import Tuple, List


def count_code_lines(file_path: str) -> Tuple[int, int]:
    """
    Count total lines and math lines in a Python source file.

    Math lines are lines containing mathematical operations:
    +, -, *, /, exp, sqrt, log, max, min, sin, cos, tan, etc.

    This function is generic and can be used for any model file.

    Args:
        file_path: Absolute path to the Python source file

    Returns:
        Tuple of (total_lines, math_lines)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    math_lines = 0

    # Keywords indicating math operations
    math_keywords = [
        'exp(', 'sqrt(', 'log(', 'max(', 'min(',
        'sin(', 'cos(', 'tan(',
        ' + ', ' - ', ' * ', ' / ',
        '+=', '-=', '*=', '/=',
        '**',
        'np.interp(',  # Interpolation (common in IR pricing)
        'discount_factor', 'forward_rate',  # IR-specific math
    ]

    for line in lines:
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Check if line contains math operations
        for kw in math_keywords:
            if kw in line:
                math_lines += 1
                break

    return total_lines, math_lines


def count_lines_in_files(file_paths: List[str]) -> Tuple[int, int]:
    """
    Count total and math lines across multiple source files.

    Args:
        file_paths: List of absolute paths to Python source files

    Returns:
        Tuple of (total_lines, math_lines) summed across all files
    """
    total = 0
    math = 0
    for path in file_paths:
        t, m = count_code_lines(path)
        total += t
        math += m
    return total, math


def format_time(seconds: float) -> str:
    """Format time in appropriate units (ms or s)."""
    if seconds < 0.1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.3f} s"


def format_number(n: int) -> str:
    """Format a large number with comma separators."""
    return f"{n:,}"
