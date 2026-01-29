#!/usr/bin/env python3
"""
Export animation-ready JSON files for the visualization dashboard.

Usage:
    python scripts/export_animation_data.py
    python scripts/export_animation_data.py --trades 50 --portfolios 5
    python scripts/export_animation_data.py --only optimization,whatif
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Export animation data for visualization")
    parser.add_argument("--trades", type=int, default=50, help="Number of trades")
    parser.add_argument("--portfolios", type=int, default=5, help="Number of portfolios/counterparties")
    parser.add_argument("--trade-types", type=str, default="ir_swap",
                        help="Comma-separated trade types (e.g. ir_swap,equity_option,fx_option)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of datasets to export (optimization,whatif,pretrade)")
    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    print(f"Exporting animation data: trades={args.trades}, portfolios={args.portfolios}, trade_types={args.trade_types}")
    if only:
        print(f"  Only: {', '.join(only)}")
    print()

    from model.json_export import export_all

    manifest = export_all(
        num_trades=args.trades,
        num_portfolios=args.portfolios,
        trade_types=args.trade_types,
        only=only,
    )

    ok_count = sum(1 for d in manifest["datasets"] if d["status"] == "ok")
    total = len(manifest["datasets"])
    print(f"\nDone: {ok_count}/{total} datasets exported successfully.")

    if ok_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
