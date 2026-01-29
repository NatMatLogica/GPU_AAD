#!/usr/bin/env python
"""
Generate publication-ready output from benchmark CSV/JSON results.

Formats:
- Markdown tables (for risk.net / quant journal / blog)
- CSV summary (for appendix)

Usage:
    # Markdown table from sweep CSVs
    python -m benchmark.export results/sweep_trades_*.csv --markdown

    # CSV summary
    python -m benchmark.export results/sweep_trades_*.csv --csv-summary

    # From JSON results
    python -m benchmark.export results/benchmark_*.json --markdown
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict


def _load_csv_rows(paths: List[Path]) -> List[Dict]:
    """Load and merge rows from multiple CSV files."""
    rows = []
    for p in paths:
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    if key in ("backend",):
                        continue
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
                rows.append(row)
    return rows


def _load_json_timing(paths: List[Path]) -> List[Dict]:
    """Extract timing rows from benchmark JSON files."""
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        config = data.get("config", {})
        timing = data.get("timing", {})
        for backend, t in timing.items():
            row = {
                "backend": backend,
                "trades": config.get("trades_per_type", 0),
                "portfolios": config.get("portfolios", 0),
                "median_ms": t.get("median_ms", t.get("p50_ms", 0)),
                "p5_ms": t.get("p5_ms", 0),
                "p95_ms": t.get("p95_ms", 0),
                "mean_ms": t.get("mean_ms", 0),
                "cv": t.get("cv", 0),
                "speedup_vs_numpy": t.get("speedup_vs_numpy", 0),
                "cost_usd": t.get("cost_usd", 0),
                "total_im": t.get("total_im", 0),
            }
            rows.append(row)
    return rows


def _format_backend_name(name: str) -> str:
    """Pretty-print backend name for publication."""
    mapping = {
        "numpy": "NumPy (ref)",
        "aadc": "AADC CPU",
        "cuda": "CUDA GPU",
        "cuda_bumpeval": "CUDA B&R",
    }
    return mapping.get(name, name)


def generate_markdown(rows: List[Dict], title: str = "SIMM Benchmark Results") -> str:
    """Generate publication-quality Markdown table."""
    lines = [
        f"## {title}",
        "",
        "| Backend | Trades | Median (ms) | P95 (ms) | Speedup | Cost ($/eval) |",
        "|---------|--------|-------------|----------|---------|---------------|",
    ]

    # Compute speedup vs first row's median if not already present
    numpy_medians = {}
    for row in rows:
        key = (row.get("trades", 0), row.get("portfolios", 0))
        if row["backend"] == "numpy":
            numpy_medians[key] = float(row.get("median_ms", 1))

    for row in rows:
        backend = _format_backend_name(str(row["backend"]))
        trades = int(float(row.get("trades", 0)))
        median = float(row.get("median_ms", 0))
        p95 = float(row.get("p95_ms", 0))
        cost = float(row.get("cost_usd", 0))

        key = (row.get("trades", 0), row.get("portfolios", 0))
        if key in numpy_medians and numpy_medians[key] > 0:
            speedup = numpy_medians[key] / median if median > 0 else 0
        else:
            speedup = float(row.get("speedup_vs_numpy", 0))

        cost_str = f"${cost:.4f}" if cost > 0 else "-"
        lines.append(
            f"| {backend:<20} | {trades:>6} | {median:>11.3f} | "
            f"{p95:>8.2f} | {speedup:>6.1f}x | {cost_str:>13} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_csv_summary(rows: List[Dict]) -> str:
    """Generate CSV summary string."""
    if not rows:
        return ""
    fields = [
        "backend", "trades", "portfolios", "median_ms",
        "p5_ms", "p95_ms", "speedup_vs_numpy", "cost_usd", "total_im",
    ]
    lines = [",".join(fields)]
    for row in rows:
        vals = []
        for f in fields:
            v = row.get(f, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Export benchmark results for publication"
    )
    parser.add_argument("files", nargs="+", help="CSV or JSON result files")
    parser.add_argument("--markdown", action="store_true",
                        help="Output Markdown table")
    parser.add_argument("--csv-summary", action="store_true",
                        help="Output CSV summary")
    parser.add_argument("--title", type=str,
                        default="ISDA-SIMM Benchmark: AADC CPU vs CUDA GPU",
                        help="Table title")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load data
    paths = [Path(f) for f in args.files]
    rows = []
    for p in paths:
        if p.suffix == ".csv":
            rows.extend(_load_csv_rows([p]))
        elif p.suffix == ".json":
            rows.extend(_load_json_timing([p]))
        else:
            print(f"Warning: unknown file type {p.suffix}, skipping {p}")

    if not rows:
        print("No data loaded.")
        sys.exit(1)

    # Default to markdown if nothing specified
    if not args.markdown and not args.csv_summary:
        args.markdown = True

    output_parts = []
    if args.markdown:
        output_parts.append(generate_markdown(rows, title=args.title))
    if args.csv_summary:
        output_parts.append(generate_csv_summary(rows))

    result = "\n\n".join(output_parts)

    if args.output:
        Path(args.output).write_text(result)
        print(f"Written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
