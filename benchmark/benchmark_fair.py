#!/usr/bin/env python
"""
Fair AADC vs CUDA SIMM Benchmark — Publication Quality.

Ensures identical inputs, identical SIMM formula, identical optimizer —
only the SIMM+gradient backend differs.

Statistical rigor:
- 95% confidence intervals (t-distribution)
- Percentiles: P5, P50, P95, P99
- Coefficient of variation with warnings
- IQR outlier exclusion

Usage:
    # Validate all backends match (no optimization)
    python -m benchmark.benchmark_fair --trades 50 --portfolios 3 --validate-only

    # Run fair benchmark with optimization
    python -m benchmark.benchmark_fair --trades 100 --portfolios 3 --optimize

    # Publication-quality run
    python -m benchmark.benchmark_fair --trades 1000 --portfolios 5 --min-runs 30 --optimize --opt-runs 3

    # CUDA simulator (no GPU)
    NUMBA_ENABLE_CUDASIM=1 python -m benchmark.benchmark_fair --trades 100 --portfolios 3

    # Full benchmark on GPU
    python -m benchmark.benchmark_fair --trades 1000 --portfolios 5 --optimize
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data_gen import generate_benchmark_data
from benchmark.backends.base import SIMMBackend
from benchmark.backends.numpy_backend import NumPyBackend
from benchmark.optimizer import optimize_allocation
from benchmark.environment import capture_environment
from benchmark.cost import CostTracker

RESULTS_DIR = Path(__file__).parent / "results"


def get_available_backends(num_threads: int = 8, num_gpus: int = 1,
                           collect_timing: bool = False) -> list:
    """Return list of available (name, backend) pairs."""
    backends = [("numpy", NumPyBackend())]

    try:
        from benchmark.backends.aadc_backend import AADCBackend, AADC_AVAILABLE
        if AADC_AVAILABLE:
            backends.append(("aadc", AADCBackend(num_threads=num_threads)))
    except ImportError:
        pass

    try:
        from benchmark.backends.cuda_backend import CUDABackend, CUDA_AVAILABLE
        if CUDA_AVAILABLE:
            backends.append(("cuda", CUDABackend(
                num_gpus=num_gpus, collect_timing=collect_timing
            )))
    except ImportError:
        pass

    try:
        from benchmark.backends.cuda_bumpeval_backend import (
            CUDABumpRevalBackend, CUDA_BUMPEVAL_AVAILABLE
        )
        if CUDA_BUMPEVAL_AVAILABLE:
            backends.append(("cuda_bumpeval", CUDABumpRevalBackend()))
    except ImportError:
        pass

    return backends


def validate_backends(backends, agg_S, factor_meta, im_tol=1e-6, grad_tol=1e-4):
    """
    Validate that all backends produce matching IM values and gradients.

    Returns True if all validations pass.
    """
    if len(backends) < 2:
        print("  Only one backend available — skipping cross-validation.")
        return True

    ref_name, ref_backend = backends[0]
    ref_im, ref_grad = ref_backend.compute_im_and_gradient(agg_S)

    all_pass = True

    for name, backend in backends[1:]:
        test_im, test_grad = backend.compute_im_and_gradient(agg_S)

        # IM validation
        im_abs_diff = np.abs(test_im - ref_im)
        im_rel_diff = im_abs_diff / np.maximum(np.abs(ref_im), 1e-10)
        im_max_rel = float(np.max(im_rel_diff))
        im_pass = im_max_rel < im_tol

        # Gradient validation — use looser tolerance for bump-and-revalue
        effective_grad_tol = 1e-2 if "bumpeval" in name else grad_tol
        grad_abs_diff = np.abs(test_grad - ref_grad)
        grad_scale = np.maximum(np.abs(ref_grad), 1e-10)
        grad_rel_diff = grad_abs_diff / grad_scale
        # Only check non-zero gradients
        nonzero_mask = np.abs(ref_grad) > 1e-10
        if np.any(nonzero_mask):
            grad_max_rel = float(np.max(grad_rel_diff[nonzero_mask]))
        else:
            grad_max_rel = 0.0
        grad_pass = grad_max_rel < effective_grad_tol

        status = "PASS" if (im_pass and grad_pass) else "FAIL"
        print(f"  {name} vs {ref_name}: "
              f"IM rel_err={im_max_rel:.2e} ({'OK' if im_pass else 'FAIL'}), "
              f"grad rel_err={grad_max_rel:.2e} ({'OK' if grad_pass else 'FAIL'}) "
              f"[{status}]")

        if not im_pass:
            print(f"    IM values ({ref_name}): {ref_im[:5]}")
            print(f"    IM values ({name}):     {test_im[:5]}")
        if not grad_pass:
            # Show worst factor
            worst_p, worst_k = np.unravel_index(
                np.argmax(grad_abs_diff), grad_abs_diff.shape
            )
            print(f"    Worst gradient: p={worst_p}, k={worst_k}, "
                  f"ref={ref_grad[worst_p, worst_k]:.6e}, "
                  f"test={test_grad[worst_p, worst_k]:.6e}")

        if not (im_pass and grad_pass):
            all_pass = False

    return all_pass


def _compute_statistics(times_ms: np.ndarray) -> dict:
    """Compute publication-quality statistics from timing array.

    Returns dict with mean, std, percentiles, CI, CV, outlier info.
    Uses IQR for outlier detection and t-distribution for CI.
    """
    n = len(times_ms)

    # IQR outlier detection
    q1 = np.percentile(times_ms, 25)
    q3 = np.percentile(times_ms, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inlier_mask = (times_ms >= lower_bound) & (times_ms <= upper_bound)
    outlier_count = int(n - np.sum(inlier_mask))
    clean = times_ms[inlier_mask]

    if len(clean) == 0:
        clean = times_ms  # fallback

    # Basic stats on clean data
    mean_ms = float(np.mean(clean))
    std_ms = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0

    # 95% CI using t-distribution
    try:
        from scipy.stats import t as t_dist
        n_clean = len(clean)
        if n_clean > 1:
            t_crit = t_dist.ppf(0.975, df=n_clean - 1)
            margin = t_crit * std_ms / np.sqrt(n_clean)
            ci_lower = mean_ms - margin
            ci_upper = mean_ms + margin
        else:
            ci_lower = ci_upper = mean_ms
    except ImportError:
        # Fallback: approximate with 1.96 * SE
        se = std_ms / np.sqrt(len(clean)) if len(clean) > 1 else 0
        ci_lower = mean_ms - 1.96 * se
        ci_upper = mean_ms + 1.96 * se

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "median_ms": float(np.median(clean)),
        "p5_ms": float(np.percentile(clean, 5)),
        "p50_ms": float(np.percentile(clean, 50)),
        "p95_ms": float(np.percentile(clean, 95)),
        "p99_ms": float(np.percentile(clean, 99)),
        "ci95_lower_ms": ci_lower,
        "ci95_upper_ms": ci_upper,
        "cv": cv,
        "outliers_excluded": outlier_count,
        "num_runs_total": n,
        "num_runs_clean": len(clean),
    }


def benchmark_backend(backend, agg_S, num_warmup=3, num_runs=30):
    """
    Benchmark a single backend's SIMM+gradient computation.

    Returns dict with timing results including statistical rigor.
    """
    # Warmup
    for _ in range(num_warmup):
        backend.compute_im_and_gradient(agg_S)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        im_vals, grads = backend.compute_im_and_gradient(agg_S)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_ms = np.array(times) * 1000.0
    stats = _compute_statistics(times_ms)

    # Warn on high CV
    if stats["cv"] > 0.05:
        print(f"    WARNING: CV={stats['cv']:.1%} > 5% — results may be noisy")

    stats["total_im"] = float(np.sum(im_vals))
    return stats


def print_benchmark_table(results):
    """Print publication-quality comparison table."""
    print(f"\n{'Backend':<15} {'Median [P5-P95] (ms)':>25} "
          f"{'Mean (ms)':>10} {'CI 95%':>18} {'CV':>6} "
          f"{'Total IM':>18} {'Speedup':>10}")
    print("-" * 110)

    ref_median = None
    for name, t in results.items():
        if ref_median is None:
            ref_median = t["median_ms"]
        speedup = ref_median / t["median_ms"] if t["median_ms"] > 0 else 0
        ci_str = f"[{t['ci95_lower_ms']:.2f}-{t['ci95_upper_ms']:.2f}]"
        p_str = f"{t['median_ms']:.3f} [{t['p5_ms']:.2f}-{t['p95_ms']:.2f}]"
        cv_str = f"{t['cv']:.1%}"
        print(f"  {name:<13} {p_str:>25} {t['mean_ms']:>10.3f} {ci_str:>18} "
              f"{cv_str:>6} ${t['total_im']:>16,.2f} {speedup:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Fair AADC vs CUDA SIMM Benchmark (Publication Quality)"
    )
    parser.add_argument("--trades", type=int, default=50,
                        help="Number of trades per type")
    parser.add_argument("--portfolios", type=int, default=3,
                        help="Number of portfolios")
    parser.add_argument("--trade-types", type=str, default="ir_swap,equity_option",
                        help="Comma-separated trade types")
    parser.add_argument("--threads", type=int, default=8,
                        help="AADC threads")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate backends match (no timing)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run allocation optimization through each backend")
    parser.add_argument("--max-iters", type=int, default=100,
                        help="Max optimization iterations")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of timing runs (legacy, use --min-runs)")
    parser.add_argument("--min-runs", type=int, default=None,
                        help="Minimum number of timing runs (default 30)")
    parser.add_argument("--opt-runs", type=int, default=1,
                        help="Number of optimization repeats (default 1)")
    parser.add_argument("--simm-buckets", type=int, default=3,
                        help="Number of currencies")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for CUDA backend")
    parser.add_argument("--cost-per-hour", type=float, default=0.0,
                        help="Cost per hour in USD (0=not tracked)")
    parser.add_argument("--platform", type=str, default="",
                        help="Platform name (e.g. hgx_h100_8gpu_onprem)")
    parser.add_argument("--collect-gpu-timing", action="store_true",
                        help="Collect H2D/kernel/D2H breakdown for CUDA")
    args = parser.parse_args()

    # Resolve num_runs: --min-runs takes priority
    num_runs = args.min_runs if args.min_runs is not None else args.num_runs
    trade_types = [t.strip() for t in args.trade_types.split(",")]
    cost_tracker = CostTracker(
        cost_per_hour=args.cost_per_hour, platform=args.platform
    )

    # Capture environment
    env = capture_environment(seed=42)

    print("=" * 70)
    print("Fair AADC vs CUDA SIMM Benchmark (Publication Quality)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Trades per type: {args.trades}")
    print(f"  Trade types:     {trade_types}")
    print(f"  Portfolios:      {args.portfolios}")
    print(f"  AADC threads:    {args.threads}")
    print(f"  Currencies:      {args.simm_buckets}")
    print(f"  Timing runs:     {num_runs}")
    print(f"  Opt runs:        {args.opt_runs}")
    print(f"  GPUs:            {args.num_gpus}")
    if args.cost_per_hour > 0:
        print(f"  Cost/hr:         ${args.cost_per_hour:.2f}")
    if args.platform:
        print(f"  Platform:        {args.platform}")
    print()

    # Step 1: Generate data (shared across all backends)
    print("Step 1: Generating benchmark data...")
    data = generate_benchmark_data(
        num_trades=args.trades,
        num_portfolios=args.portfolios,
        trade_types=trade_types,
        num_simm_buckets=args.simm_buckets,
        num_threads=args.threads,
    )

    # Step 2: Set up backends
    print("\nStep 2: Setting up backends...")
    backend_pairs = get_available_backends(
        num_threads=args.threads,
        num_gpus=args.num_gpus,
        collect_timing=args.collect_gpu_timing,
    )
    active_backends = []
    for name, backend in backend_pairs:
        try:
            backend.setup(data.factor_meta)
            active_backends.append((name, backend))
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not active_backends:
        print("No backends available!")
        sys.exit(1)

    # Build aggregated sensitivities for validation/benchmarking
    agg_S = (data.S.T @ data.initial_allocation).T  # (P, K)

    # Step 3: Validate
    print("\nStep 3: Validating backend consistency...")
    valid = validate_backends(active_backends, agg_S, data.factor_meta)
    if not valid:
        print("\nWARNING: Backend validation FAILED. Results may not be comparable.")
    else:
        print("\n  All backends produce consistent results.")

    if args.validate_only:
        print("\n--validate-only specified. Done.")
        return

    # Step 4: Benchmark SIMM+gradient computation
    print(f"\nStep 4: Benchmarking SIMM+gradient ({num_runs} runs each)...")
    timing_results = {}
    for name, backend in active_backends:
        print(f"  Benchmarking {name}...")
        timing_results[name] = benchmark_backend(
            backend, agg_S, num_warmup=3, num_runs=num_runs
        )
        # Add cost if configured
        if cost_tracker.cost_per_hour > 0:
            median_s = timing_results[name]["median_ms"] / 1000.0
            timing_results[name]["cost_usd"] = cost_tracker.compute(median_s)

    print_benchmark_table(timing_results)

    # Step 5: Optimization (if requested)
    opt_results = {}
    if args.optimize:
        print(f"\nStep 5: Running optimization ({args.max_iters} max iters, "
              f"{args.opt_runs} repeat(s))...")
        for name, backend in active_backends:
            print(f"\n  --- {name} backend ---")
            run_results = []
            for run_idx in range(args.opt_runs):
                if args.opt_runs > 1:
                    print(f"    [Run {run_idx + 1}/{args.opt_runs}]")
                result = optimize_allocation(
                    backend=backend,
                    S=data.S,
                    initial_allocation=data.initial_allocation,
                    max_iters=args.max_iters,
                    verbose=(args.opt_runs == 1),
                )
                run_results.append(result)
                if args.opt_runs > 1:
                    print(f"      IM ${result.initial_im:,.2f} -> ${result.final_im:,.2f} "
                          f"({result.num_iterations} iters, "
                          f"{result.total_time*1000:.1f}ms)")

            # Aggregate across runs
            median_total_time = float(np.median([r.total_time for r in run_results]))
            median_final_im = float(np.median([r.final_im for r in run_results]))
            median_iters = int(np.median([r.num_iterations for r in run_results]))

            opt_results[name] = {
                "runs": run_results,
                "median_total_time_ms": median_total_time * 1000,
                "median_final_im": median_final_im,
                "median_iterations": median_iters,
            }

            best = run_results[0]
            print(f"  Result: IM ${best.initial_im:,.2f} -> ${median_final_im:,.2f} "
                  f"(median {median_iters} iters, median {median_total_time*1000:.1f}ms)")

        # Compare optimization results
        print(f"\n{'Backend':<15} {'Initial IM':>16} {'Final IM':>16} "
              f"{'Reduction':>10} {'Moves':>6} {'Iters':>6} "
              f"{'Eval (ms)':>10} {'Total (ms)':>10}")
        print("-" * 95)
        for name, od in opt_results.items():
            best = od["runs"][0]
            reduction = (1 - od["median_final_im"] / best.initial_im) * 100 if best.initial_im > 0 else 0
            print(f"  {name:<13} ${best.initial_im:>14,.2f} ${od['median_final_im']:>14,.2f} "
                  f"{reduction:>9.2f}% {best.trades_moved:>6} "
                  f"{od['median_iterations']:>6} "
                  f"{best.eval_time*1000:>10.1f} {od['median_total_time_ms']:>10.1f}")

    # Step 6: Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"benchmark_{timestamp}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "environment": env.to_dict(),
        "cli_args": " ".join(sys.argv),
        "config": {
            "trades_per_type": args.trades,
            "trade_types": trade_types,
            "portfolios": args.portfolios,
            "num_factors": data.num_factors,
            "num_trades": data.num_trades,
            "num_runs": num_runs,
            "opt_runs": args.opt_runs,
            "num_gpus": args.num_gpus,
        },
        "cost": cost_tracker.to_dict(),
        "timing": timing_results,
        "validation_passed": valid,
    }
    if args.optimize:
        output["optimization"] = {
            name: {
                "initial_im": od["runs"][0].initial_im,
                "median_final_im": od["median_final_im"],
                "median_iterations": od["median_iterations"],
                "median_total_time_ms": od["median_total_time_ms"],
                "trades_moved": od["runs"][0].trades_moved,
                "im_history": [float(x) for x in od["runs"][0].im_history],
            }
            for name, od in opt_results.items()
        }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {result_file}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
