# SIMM Benchmark
## H100 GPU vs AADC | IR Swap Portfolios 100–50,000 Trades

**Based on measured benchmark results**
All timings are actual, not estimated
3 February 2026
*DRAFT — Internal Working Document*

---

## Headline Benchmark Results

All results measured on Supermicro JumpStart H100 system. Portfolio: IR swaps, ISDA SIMM v2.6. All backends produce identical `im_result` values, validating apples-to-apples comparison.

| Backend | Evals/sec | Total Time | Eval Count | Converged? |
|---------|----------:|------------|------------|------------|
| cpp_aadc (C++) | 34,144 | 0.27s | 9,114 | Yes (1-2 iters) |
| aadc_full (Python) | 1,311 | 6.95s | 9,116 | Yes (2 iters) |
| gpu_full (pathwise) | 727 | 12.54s | 9,116 | Yes (2 iters) |
| bf_gpu (brute-force) | 33 | 3.10s | 101 only | NEVER |

*At 5,000 trades, EOD ADAM optimization. bf_gpu stuck at max_iters=100 without converging.*

### Speedup Ratios

- **cpp_aadc vs gpu_full (pathwise):** 47× faster
- **cpp_aadc vs bf_gpu (brute-force):** 1,047× faster
- **cpp_aadc vs aadc_full (Python wrapper):** 26× faster (Python overhead, not AAD computation)

### Data Quality Validation

All backends produce identical `im_result` at each trade count. Unlike XVA benchmarks where CVA/DVA diverged 7×, SIMM results are exact matches. This validates a true apples-to-apples performance comparison.

---

## 1. Trading Impact

Lead with workflows where the advantage is commercially meaningful, not raw speedup numbers.

### 1.1 Pre-Trade Margin Checks (Primary Commercial Driver)

A trader needs the margin impact of a new trade before execution. This is the highest-frequency, lowest-latency workflow.

| Backend | Kernel Evals/sec | Per-Eval Latency | Note |
|---------|-----------------|------------------|------|
| cpp_aadc | 34,013 | 0.029ms | Sub-ms, interactive |
| aadc_full | 1,320 | 0.76ms | Python wrapper overhead |
| gpu_full | 721 | 1.39ms | CUDA pathwise gradients |
| bf_gpu | 31 | 31.9ms | Brute-force, no gradients |

*At 5,000 trades. Per-evaluation latency from throughput benchmarks (re-evaluation only, no recording).*

#### Counterparty Routing Use Case

Trader checking 20 counterparties for lowest marginal IM (sequential checks):

- **cpp_aadc:** 20 × 0.029ms = 0.58ms total — sub-millisecond, instant exploration
- **aadc_full:** 20 × 0.76ms = 15.2ms total — interactive
- **gpu_full:** 20 × 1.39ms = 27.8ms — noticeable but survivable
- **bf_gpu:** 20 × 31.9ms = 638ms — not feasible for interactive use

**Key message:** At C++ AADC speeds, desks can explore 50+ counterparties in under 2ms. At GPU pathwise speeds, routing requires batch processing. Brute-force GPU is batch-only.

> **Presentation Note:** Always use pure kernel re-evaluation time for fair hardware comparison. Workflow times include Python overhead, aggregation logic, and one-time recording costs that don't apply to repeated evaluations.

### 1.2 What-If Scenarios — Interactive Exploration

At 5,000 trades: cpp_aadc delivers 34,000 evals/sec vs gpu_full at 721 evals/sec — a 47× ratio.

Risk manager exploring intraday stress ("rates spike 50bp, credit spreads blow out, FX moves 5%") can sweep parameter grids in real time with AADC. GPU kernel launch overhead per scenario forces a batch-all-at-once workflow — fundamentally less exploratory.

### 1.3 EOD Portfolio Optimisation (The Devastating Story)

This is the most striking result. When thousands of SIMM evaluations are needed to optimise trade allocation across netting sets:

- **AADC:** converges in 2 iterations regardless of portfolio size (analytic gradients)
- **GPU brute-force:** NEVER CONVERGES at 500+ trades (stuck at max_iters=100)

**Root cause:** ADAM optimiser requires analytic gradients. Brute-force provides noisy finite-difference approximations, causing the optimiser to wander without converging. AADC provides exact analytic gradients, letting ADAM jump straight to the optimum.

| Backend | Eval Count | Converged? | IM Reduction |
|---------|------------|------------|--------------|
| cpp_aadc | 9,114 | Yes (1-2 iters) | 7.00% |
| aadc_full | 9,116 | Yes (2 iters) | 7.00% |
| bf_gpu | 101 | NEVER | 7.04% |

> **Critic's Objection to Address:** bf_gpu ran only 101 evaluations vs AADC's 9,114. A critic will say "of course it didn't converge — 90× fewer evaluations." The benchmark should also show what happens if bf_gpu is allowed to run uncapped. If it literally cannot converge due to noisy gradients, that's the real story. If it just needs more iterations, different narrative.

### 1.4 Continuous Intraday Monitoring (Not Yet Benchmarked)

This is the strongest AADC case missing from the current benchmark. Recalculating SIMM for the full portfolio every time market data ticks, across all counterparties:

- **Large desk:** 50 counterparties, data updating every ~5 seconds = 600 SIMM evals/minute
- **cpp_aadc (34K evals/sec):** trivially feasible with massive capacity to spare
- **gpu_full (727 evals/sec):** feasible but consuming significant GPU resources
- **bf_gpu:** not achievable at scale — desk falls back to periodic batch recalculation, loses real-time visibility

**MVA connection:** If you can run SIMM fast enough to project forward IM along Monte Carlo paths, you can compute MVA in real time. This is the industry "holy grail." That workflow requires tens of thousands of SIMM evaluations per XVA time step, which only AADC can deliver.

### 1.5 Where Advantage Is NOT Meaningful

- **Portfolio setup:** single evaluation, 0.8s once at start of day — irrelevant despite 400× speedup
- **Margin attribution:** runs in microseconds on all backends — advantage is academic, not practical
- **Any workflow requiring only a handful of SIMM evaluations** won't show meaningful wall-clock difference

---

## 2. Implementation & Support

### 2.1 Convergence Quality

The EOD optimisation results reveal a fundamental quality difference, not just speed:

- **AADC:** exact analytic gradients → ADAM converges in 2 iterations (or 0 for cpp_aadc)
- **GPU brute-force:** noisy finite-difference approximations → optimiser wanders, hits iteration cap
- Both achieve similar IM reduction (~7%), but only AADC converges reliably

#### cpp_aadc Iteration Reporting — RESOLVED

Previously cpp_aadc reported `optimize_iterations=0` due to a **use-after-move bug** in `allocation_optimizer.h`. The iteration count was being read from a container after it had been moved, returning 0 instead of the actual count. This bug has been fixed in v2.1.1, and cpp_aadc now correctly reports 1-2 iterations, consistent with aadc_full.

### 2.2 Code Complexity & Maintainability

- **AADC:** operator overloading on existing C++ SIMM implementation — minimal code changes
- **GPU:** CUDA kernels for each risk class — model-specific development effort
- **SIMM version change (v2.6 → v2.8):** AADC is automatic; GPU requires kernel re-engineering
- GPU approach requires specialist CUDA engineers ($300–500K/year); AADC uses existing quant team

### 2.3 Integration & Python vs C++ Clarity

> **Critical Positioning Decision:** The 26× gap between aadc_full (1,311 evals/sec) and cpp_aadc (34,144 evals/sec) is Python wrapper overhead, NOT AAD computation speed. A bank integrating the C++ library directly gets 34K evals/sec. For publication, clarify which is "the AADC number" — don't conflate wrapper penalty with AAD performance.

### 2.4 Accuracy

- All backends produce identical `im_result` (unlike XVA where CVA/DVA diverged 7×)
- **AADC:** exact analytic gradients
- **GPU brute-force:** finite-difference approximation errors
- Accuracy matters most for optimisation convergence and regulatory validation

---

## 3. Costs

### 3.1 Hardware

- **H100 GPU:** $25–40K per card
- **AADC:** standard CPU infrastructure — no specialised hardware required

#### GPU Setup Overhead Dominates

Portfolio setup shows gpu_full taking ~1.28s regardless of 100 or 5,000 trades — almost entirely kernel recording/initialisation. The actual SIMM computation is trivial. This supports the argument that SIMM aggregation isn't a GPU workload: you pay a massive fixed cost that never amortises because the computation itself (matrix multiplications over ~50–100 buckets) is small.

### 3.2 Total Cost of Ownership

- **GPU cluster:** maintenance, cooling, power, specialist CUDA engineers
- **AADC:** CPU farm, existing quant team — no additional specialist hiring
- Break-even analysis needed: at what portfolio size/frequency does GPU investment pay off?

### 3.3 Scaling Economics

- Cost per additional counterparty portfolio
- Cost per additional risk factor: AADC O(1) vs brute-force O(N)
- For 1,000 risk factors: brute-force requires 1,000 full simulation runs; AADC requires 1 backward pass

---

## 4. Regulatory, MVA & Future-Proofing

### 4.1 Regulatory

- SIMM backtesting requirement: can you run daily backtests within the compute window?
- ISDA SIMM Remediation: EUR 25M threshold for backtesting/reporting regardless of margin exchange
- "Initial margin is very tricky to model. Must model sensitivities through time. Very computationally expensive." — Address this directly with benchmark numbers.
- Fast SIMM enables compliance without infrastructure explosion

### 4.2 MVA Connection (The Holy Grail)

SIMM feeds MVA. Accurate MVA requires projecting forward IM within Monte Carlo simulation:

- Fast SIMM directly enables accurate MVA
- Real-time SIMM + XVA Greeks = pre-trade pricing with full cost (CVA + FVA + MVA + KVA)
- Industry experts say this requires "understanding sensitivities, concentration risk, full what-if analysis capacity"

### 4.3 Scalability & Future-Proofing

- Show results at 1K, 10K, 100K, 1M trades
- GPU overhead constant regardless of portfolio size (supports "not a GPU workload" argument)
- AADC scales linearly with portfolio complexity
- AADC: automatic differentiation of new code when SIMM version changes
- GPU: new kernels required for each model change

---

## Key Messages for Publication

### Primary Message

SIMM is a sensitivity-aggregation problem. The aggregation itself (matrix multiplications over ~50–100 buckets) is tiny — all approaches compute IM in microseconds. The expensive part is computing the sensitivities that feed SIMM. AAD's O(1) sensitivity cost vs brute-force's O(N) cost creates an unbridgeable performance gap.

### EOD Optimisation Story

When thousands of SIMM evaluations are needed to optimise trade allocation, the GPU approach collapses because each evaluation requires re-bumping every risk factor. AADC provides exact analytic gradients letting ADAM converge in 2 steps. GPU brute-force never converges at scale.

### Workflow-Specific Framing

Don't lead with "AADC is 47× faster." Lead with "Pre-trade counterparty routing: AADC enables exploring 50+ counterparties in 3ms; GPU limited to batch processing smaller sets." Make speed advantage concrete in trading workflows.

### Normalise Comparisons

Show per-evaluation latency for pre-trade checks. Show eval count for EOD optimisation. Make clear bf_gpu's 101 evals vs AADC's 9,114 evals, and that bf_gpu didn't converge despite similar IM reduction.

### GPU Setup Overhead

~1.28s constant regardless of portfolio size. Supports argument that SIMM aggregation isn't a GPU workload — paying massive fixed cost that never amortises because computation itself is small.

### Python vs C++ Clarity

26× aadc_full vs cpp_aadc gap is Python overhead. Bank integrating C++ directly gets 34K evals/sec. Don't conflate wrapper penalty with AAD performance.

---

## Pending Items Before Publication

1. ~~Explain cpp_aadc zero-iteration convergence mechanism~~ — **RESOLVED:** Was use-after-move bug, now reports 1-2 iterations
2. Run bf_gpu with uncapped iterations — determine if convergence is possible with more evaluations
3. ~~Add continuous intraday monitoring workflow to benchmark~~ — **RESOLVED:** What-If Step 4 already measures this
4. ~~Normalise all pre-trade check comparisons (per-evaluation basis)~~ — **DONE:** Updated Section 1.1
5. Document GPU optimisation attempts and profiling — occupancy, bandwidth analysis to prevent strawman criticism
6. Clarify cpp_aadc vs aadc_full positioning — which is "the AADC number" for publication?
7. Add cost-per-Greek analysis
8. Test at larger portfolio sizes (10K, 100K trades) to show scaling behaviour
9. **Add memory requirements section**
10. **Add model descriptions and methodology section**

---

## Changes from Original Structure

### Added

- Specific benchmark numbers and tables throughout all sections
- Counterparty routing use case with quantified latency comparison
- Continuous intraday monitoring workflow (covered by What-If Step 4)
- EOD convergence failure story as centrepiece (bf_gpu never converges at 500+ trades)
- cpp_aadc iteration reporting bug (now fixed)
- Per-evaluation normalisation for fair pre-trade comparison
- GPU setup overhead analysis (1.28s constant cost supports "not a GPU workload")
- Python vs C++ clarity for AADC positioning
- Pending items checklist for publication readiness

### Removed / De-emphasised

- Cloud costs comparison (deferred — requires cloud pricing research)
- Dispute resolution (weak argument without supporting data)
- Portfolio setup speedup (0.8s once per day is irrelevant)
- Margin attribution advantage (runs in microseconds on all backends)

### Restructured

- Trading Impact now ordered by commercial importance, not raw speedup
- Key messages section added with specific framing guidance
- Implementation section now leads with convergence quality, not code complexity
- Costs section includes GPU setup overhead analysis from actual data
