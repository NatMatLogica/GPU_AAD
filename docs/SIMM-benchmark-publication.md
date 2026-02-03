# SIMM Benchmark Publication Structure

**REVISED — Corrected Framing & Clarity Pass**

## AADC (Adjoint Algorithmic Differentiation Compiler) vs H100 GPU Backends for ISDA SIMM Workflows

*February 2026 — Based on Supermicro JumpStart H100 Benchmarks*

---

## Framing Corrections — What Changed in This Revision

The initial draft contained framing that could mislead readers (including AI summarisers) about what MatLogica provides, what this benchmark measures, and what SIMM actually is. This section documents the corrections.

### Correction 1: MatLogica Does NOT Provide a SIMM Calculator

**What readers/AI got wrong:**
The initial framing read as though MatLogica sells a SIMM product — the AI-generated infographic described it as comparing "AADC against H100 GPU backends for ISDA SIMM v2.6 calculations" and positioned AADC as a SIMM engine.

**What MatLogica actually provides:** AADC is a C++ compiler-level automatic differentiation library. It transforms any numerical code into code that also computes exact derivatives (sensitivities/gradients) via operator overloading. Banks apply AADC to their own existing pricing and risk code — including their own SIMM implementations.

The benchmark uses an ISDA SIMM implementation as the test vehicle, but the product being benchmarked is the differentiation engine underneath, not the SIMM model on top. The correct framing is:

> "This benchmark measures the performance of AADC — an automatic differentiation library — applied to a bank's own SIMM code, compared against GPU-based approaches to the same SIMM workflows."

### Correction 2: SIMM Is NOT a "Sensitivity-Aggregation Problem"

**What readers/AI got wrong:**
The initial draft stated "SIMM is a sensitivity-aggregation problem" as the primary message. An AI reader picked this up and built the entire infographic around it. This is misleading: it describes what SIMM does in isolation, but not what the benchmark is actually measuring.

**What SIMM does:** takes pre-computed CRIF sensitivities as input and aggregates them through the ISDA formula (weighted sums, correlations, concentration thresholds across risk classes). The aggregation step itself is a small matrix operation — microseconds on any hardware.

**What this benchmark actually measures:** the end-to-end cost of SIMM-dependent workflows that require computing sensitivities, evaluating SIMM, and in the case of optimisation, differentiating through SIMM repeatedly. Specifically:

- **Pre-trade checks:** compute sensitivities for a candidate trade, evaluate SIMM to get margin impact
- **What-if scenarios:** re-compute sensitivities under stressed market data, re-evaluate SIMM
- **EOD optimisation:** find the trade allocation across netting sets that minimises total IM. This requires thousands of SIMM evaluations PLUS exact gradients of IM with respect to allocation decisions

The critical insight is not that aggregation is small (it is), but that AADC's kernel recording and replay mechanism makes the entire sensitivity-then-SIMM pipeline dramatically faster, and AADC's analytic gradients make optimisation converge where finite-difference gradients fail.

### Correction 3: Clarify "Backends" — These Are Not Interchangeable SIMM Engines

**What readers/AI could get wrong:**
The benchmark labels (cpp_aadc, aadc_full, gpu_full, bf_gpu, pure_gpu_ir) look like competing SIMM products. They are not. They are different computational approaches to the same underlying SIMM code.

**What each backend actually is:**

| Backend | Description |
|---------|-------------|
| **cpp_aadc** | AADC's C++ kernel — records the computation graph once, replays it with new inputs. Direct C++ integration (no Python layer). |
| **aadc_full** | Same AADC C++ kernel, accessed through a Python wrapper. The 26× slower performance vs cpp_aadc is pure Python marshalling overhead. |
| **gpu_full** (pathwise) | H100 GPU computing sensitivities via pathwise differentiation through pre-recorded CUDA graph. Supports analytic gradients. |
| **bf_gpu** (brute-force) | H100 GPU computing sensitivities by bumping each risk factor and re-evaluating. No analytic gradients — only finite-difference approximations. |
| **pure_gpu_ir** | Native GPU implementation of a subset of the IR risk class only. |

The comparison is between computation strategies, not products. A bank choosing AADC would apply it to their existing SIMM code. A bank choosing the GPU approach would rewrite their SIMM pipeline in CUDA.

### Correction 4: Other Potential Misreadings

**4a. "34,144 evaluations per second" does not mean AADC evaluates 34K trades**

An "evaluation" is one complete pass of the SIMM formula over the entire portfolio (all trades, all risk classes, all netting sets). At 5,000 trades, cpp_aadc completes 34,144 such full-portfolio SIMM evaluations per second. This is the throughput needed for optimisation loops, not the number of trades processed.

**4b. "Converged in 0 iterations" was a bug — now fixed**

~~cpp_aadc reports optimize_iterations=0 yet optimize_converged=True with the same IM reduction as aadc_full (which reports 2 iterations). This likely means cpp_aadc computes exact analytic gradients and reaches the optimum without iterative search.~~

**UPDATE:** This was a use-after-move bug in `allocation_optimizer.h` (v2.1.0). The iteration count was read from a container after it had been moved, returning 0. Fixed in v2.1.1 — cpp_aadc now correctly reports 1-2 iterations, consistent with aadc_full.

**4c. bf_gpu's 3.10s wall-clock is misleading vs AADC's 0.27s**

bf_gpu ran only 101 evaluations (stuck at max_iters=100) in 3.10s. AADC ran 9,114 evaluations in 0.27s. The 101-vs-9114 eval count difference must be shown alongside wall-clock times. bf_gpu appears "faster" in total time only because it gave up 90× earlier — and still didn't converge.

**UPDATE:** With uncapped iterations (max_iters=3000), bf_gpu converges at **1,848 iterations** taking ~100 seconds at T=5000. It CAN converge, but requires 18× more iterations and 370× more wall-clock time than AADC.

**4d. "AADC runs on standard CPU" does not mean any CPU is sufficient**

AADC's kernel recording/replay is CPU-based but benefits from large caches and high single-thread performance. The benchmark used the CPU cores in the Supermicro H100 system (16 threads). Production deployments would use comparable server-grade CPUs — not a laptop.

**4e. "All backends produce identical im_result" needs qualification**

The im_result (IM value) is identical because the SIMM aggregation formula is deterministic given the same sensitivities. However, the sensitivity computation methods differ, and for more complex instruments (beyond IR swaps) the sensitivity values themselves may diverge. The benchmark uses IR swaps specifically because sensitivities are analytically tractable on all backends.

**4f. Do not conflate aadc_full and cpp_aadc as "the AADC number"**

aadc_full (1,311 evals/sec) includes Python marshalling overhead. cpp_aadc (34,144 evals/sec) is the raw library performance. A bank integrating the C++ library gets cpp_aadc speeds. A bank using the Python API gets aadc_full speeds. Every table should label both clearly, and the headline numbers should specify which variant is being cited.

---

## Headline Benchmark Results

All results measured on Supermicro JumpStart H100 system. Portfolio: IR swaps, ISDA SIMM v2.6. All backends produce identical im_result values for each portfolio size, validating apples-to-apples comparison of the SIMM aggregation step. AADC is MatLogica's automatic differentiation library applied to the SIMM code — not a SIMM product.

### EOD Optimisation — 5,000 Trades (ADAM Optimiser)

| Backend | Evals/sec | Total Time | SIMM Evals | Converged? |
|---------|----------:|------------|------------|------------|
| cpp_aadc (AADC C++) | 34,144 | 0.27s | 9,114 | Yes (1-2 iters) |
| aadc_full (AADC Python) | 1,311 | 6.95s | 9,116 | Yes (2 iters) |
| gpu_full (GPU pathwise) | 727 | 12.54s | 9,116 | Yes (2 iters) |
| bf_gpu (GPU brute-force) | 33 | 3.10s | 101 | No (at 100 iters) |

*bf_gpu stuck at max_iters=100 without converging. Its 3.10s wall-clock covers only 101 evaluations vs AADC's 9,114.*

**With uncapped iterations:** bf_gpu converges at **1,848 iterations** in ~100 seconds — 370× slower than cpp_aadc.

### Speedup Ratios

- **cpp_aadc vs gpu_full (pathwise):** 47× faster per evaluation
- **cpp_aadc vs bf_gpu (brute-force):** 1,047× faster per evaluation
- **cpp_aadc vs aadc_full (Python wrapper):** 26× — this is Python overhead, not AAD computation difference

### Key Insight

The 47× speedup over GPU pathwise comes from AADC's kernel recording mechanism: record the computation graph once, replay with different inputs at minimal cost. The 1,047× over brute-force reflects the additional cost of finite-difference sensitivity computation (bumping each risk factor individually).

---

## 1. Trading Impact

Present workflows in order of commercial importance. Lead with the trading use case, not the speedup number. Make the advantage concrete in business terms.

### 1.1 Pre-Trade Margin Checks (Primary Commercial Driver)

A trader needs the margin impact of a new trade before execution. At the point of trade, the question is: "how much additional IM will this trade require, and at which counterparty is the marginal IM lowest?" AADC answers this in sub-millisecond time.

#### Pure Kernel Evaluation Latency (Re-evaluation Only)

| Backend | Kernel Evals/sec | Per-Eval Latency | Note |
|---------|-----------------|------------------|------|
| cpp_aadc (AADC C++) | 34,013 | **0.029 ms** | Sub-ms, instant |
| aadc_full (AADC Python) | 1,320 | 0.76 ms | Python wrapper overhead |
| gpu_full (GPU pathwise) | 721 | 1.39 ms | CUDA pathwise gradients |
| bf_gpu (GPU brute-force) | 31 | 31.9 ms | No analytic gradients |

*At 5,000 trades. Per-evaluation latency from throughput benchmarks (re-evaluation only, excludes one-time kernel recording).*

#### Counterparty Routing — The Business Case

Trader checking 20 counterparties for lowest marginal IM (sequential checks):

- **cpp_aadc:** 20 × 0.029ms = **0.58ms total** — sub-millisecond, instant exploration
- **aadc_full:** 20 × 0.76ms = 15.2ms — interactive
- **gpu_full:** 20 × 1.39ms = 27.8ms — noticeable but survivable
- **bf_gpu:** 20 × 31.9ms = 638ms — not feasible for interactive use

At AADC C++ speeds, desks can explore 50+ counterparties in under 2ms. At GPU pathwise speeds, the workflow is still interactive but noticeably slower. At brute-force GPU speeds, the workflow becomes batch-oriented. This constrains counterparty optimisation and may leave margin savings on the table.

> **Normalisation Required for Publication:** Always use pure kernel re-evaluation time for fair hardware comparison. Workflow times include Python overhead, aggregation logic, and one-time recording costs that don't apply to repeated evaluations.

### 1.2 What-If Scenarios — Interactive Exploration

At 5,000 trades: cpp_aadc delivers **34,000 evals/sec** vs gpu_full at **721 evals/sec** — a **47× ratio**.

Risk manager exploring intraday stress ("rates spike 50bp, credit spreads blow out, FX moves 5%") can sweep parameter grids in real time with AADC. GPU kernel launch overhead per scenario forces a batch-all-at-once workflow — fundamentally less exploratory.

**Why this matters:** Interactive what-if analysis lets risk managers discover non-obvious margin concentrations and identify rebalancing opportunities before they become urgent. Batch workflows answer pre-defined questions; interactive workflows enable discovery.

### 1.3 EOD Portfolio Optimisation (The Convergence Story)

This is the most striking result. When thousands of SIMM evaluations are needed to optimise trade allocation across netting sets:

| Trades | Backend | SIMM Evals | Total Time | IM Reduction | Converged? |
|--------|---------|------------|------------|--------------|------------|
| 500 | cpp_aadc | 437 | 49ms | 20.28% | Yes (1-2 iters) |
| 500 | bf_gpu | 101 | 279ms | 20.80% | No (at 100) |
| 500 | bf_gpu | **170** | 436ms | 20.80% | **Yes** (uncapped) |
| 5000 | cpp_aadc | 9,114 | 267ms | 7.00% | Yes (1-2 iters) |
| 5000 | bf_gpu | 101 | 3.10s | 7.04% | No (at 100) |
| 5000 | bf_gpu | **1,848** | ~100s | 7.20% | **Yes** (uncapped) |

**Root cause:** The ADAM optimiser benefits from exact gradients. AADC provides these via automatic differentiation. Brute-force GPU approximates gradients via finite differences, which are noisier. With default max_iters=100, bf_gpu doesn't have enough iterations to converge at scale. With uncapped iterations, bf_gpu CAN converge — but takes **18× more iterations** and **370× more wall-clock time** than AADC.

**The real story:** It's not that bf_gpu "cannot converge" — it's that convergence requires dramatically more compute time, making it impractical for production EOD workflows where the compute window is limited.

### 1.4 Continuous Intraday Monitoring

Recalculating SIMM for the full portfolio every time market data ticks, across all counterparties:

- 50 counterparties × market data ticks every 5 seconds = 600 SIMM evals/minute
- **cpp_aadc (34K evals/sec):** trivially feasible with capacity to spare
- **gpu_full (721 evals/sec):** feasible but consuming significant GPU resources for SIMM alone
- **bf_gpu:** not achievable within the time window; desk falls back to periodic batch, losing real-time visibility

**Note:** The What-If benchmark (Step 4) already measures this workflow — applying market shocks to aggregated sensitivities and recomputing IM for all portfolios.

### 1.5 Where Advantage Is NOT Meaningful — De-Emphasise in Publication

- **Portfolio setup:** 0.8s once at start of day is irrelevant, despite 400× speedup in relative terms
- **Margin attribution:** microseconds on all backends; AADC advantage is academic
- **Any workflow needing fewer than ~10 SIMM evaluations:** wall-clock difference too small to matter

---

## 2. Implementation & Support

### 2.1 Convergence Quality

The EOD optimisation results reveal a fundamental quality difference, not just speed:

- **AADC (both cpp_aadc and aadc_full):** exact analytic gradients via automatic differentiation → ADAM converges in 1-2 iterations
- **gpu_full (pathwise):** also provides analytic gradients, also converges in 2 iterations — but at 47× lower throughput
- **bf_gpu (brute-force):** finite-difference gradient approximations → noisier signal → requires 18× more iterations to converge

### 2.2 What AADC Integration Looks Like

AADC is a C++ library using operator overloading. Integration does not require rewriting the SIMM code:

- Bank's existing C++ SIMM implementation is instrumented with AADC types (replacing `double` with `aadc::idouble`)
- AADC records the computation graph on first execution, then replays with new inputs at minimal cost
- No CUDA development, no GPU kernel engineering, no hardware-specific optimisation
- SIMM version changes (e.g., v2.6 → v2.8): recompile with AADC, graph is automatically updated

**GPU approach by contrast requires:**

- CUDA kernel development for each risk class and model type
- Specialist GPU engineers ($300–500K/year) for development and maintenance
- Re-engineering when SIMM version changes or new instrument types are added

### 2.3 Python vs C++ Performance Clarity

> **Presentation Risk — Which Is "The AADC Number"?**
>
> aadc_full (1,311 evals/sec via Python wrapper) is 26× slower than cpp_aadc (34,144 evals/sec via direct C++). This gap is pure Python marshalling overhead, not AAD performance. A bank integrating the C++ library gets 34K evals/sec; a bank using the Python API gets 1.3K. Every table and chart must label both variants. Headline numbers should specify "AADC C++ (cpp_aadc)" when citing 34K.

### 2.4 Accuracy

- All backends produce identical im_result for each portfolio size (unlike XVA where CVA/DVA diverged 7×)
- This validates apples-to-apples comparison of the SIMM aggregation step specifically
- **AADC:** exact analytic gradients for the optimiser
- **GPU brute-force:** finite-difference approximation errors compound in optimisation loops
- **Note:** sensitivity values for more complex instruments (beyond IR swaps) may diverge between methods — this benchmark uses IR swaps because sensitivities are analytically tractable on all backends

---

## 3. Costs

### 3.1 Hardware

- **H100 GPU:** $25–40K per card, plus PCIe/NVLink infrastructure
- **AADC:** runs on standard server-grade CPUs — no GPU hardware required for SIMM workflows

**GPU Setup Overhead Supports the Cost Story:**
gpu_full takes ~1.28s for portfolio setup regardless of 100 or 5,000 trades. This constant cost is almost entirely CUDA kernel recording/initialisation. The actual SIMM computation is trivial. This means GPU approaches pay a massive fixed cost that never amortises — because SIMM aggregation (weighted sums over ~50–100 buckets per risk class) simply isn't large enough to justify GPU kernel overhead.

### 3.2 Total Cost of Ownership

- **GPU cluster:** hardware + cooling + power + specialist CUDA engineers ($300–500K/year each)
- **AADC on CPU:** standard server infrastructure + existing quant development team
- No additional specialist hiring — quants instrument their own code with AADC types

### 3.3 Scaling Economics

- **Cost per additional counterparty portfolio:** AADC adds 0.029ms per counterparty; GPU pathwise adds 1.39ms; bf_gpu adds 31.9ms
- **Cost per additional risk factor:** AADC O(1) backward pass vs brute-force O(N) bump-and-revalue
- For a desk with 1,000 risk factors: brute-force needs 1,000 re-evaluations; AADC needs 1 backward pass

---

## 4. Regulatory, MVA & Future-Proofing

### 4.1 Regulatory

- **SIMM backtesting:** can you run daily backtests within the compute window? At AADC speeds, comfortably yes.
- **ISDA SIMM Remediation:** EUR 25M threshold for backtesting/reporting regardless of margin exchange
- "Initial margin is very tricky to model. Must model sensitivities through time. Very computationally expensive." — AADC benchmark numbers address this directly.
- Fast SIMM enables compliance without infrastructure explosion

### 4.2 MVA Connection

SIMM feeds MVA (Margin Valuation Adjustment). Accurate MVA requires projecting forward IM within Monte Carlo simulation:

- Each MC path requires SIMM evaluation at each time step
- 1,000 paths × 20 time steps = 20,000 SIMM evaluations per MVA computation
- Only AADC-class throughput (34K evals/sec) makes real-time MVA feasible
- This connects SIMM performance directly to pre-trade pricing with full cost transparency (CVA + FVA + MVA + KVA)

### 4.3 Scalability & Future-Proofing

- Show benchmark results at 1K, 10K, 100K, 1M trades to demonstrate scaling behaviour
- GPU setup overhead remains constant regardless of portfolio size — supports "not a GPU workload" argument
- **AADC:** recompile when pricing models change; computation graph updates automatically
- **GPU:** each model change requires CUDA kernel re-engineering

---

## Key Messages for Publication

### Primary Message (Corrected)

**Don't say:** "SIMM is a sensitivity-aggregation problem where AADC's efficiency outperforms GPU hardware."

**Do say:** "SIMM workflows — pre-trade margin checks, what-if analysis, and portfolio optimisation — require repeated SIMM evaluation with exact gradients. AADC's kernel recording delivers 47× faster throughput than GPU pathwise computation, and its analytic gradients enable optimisation convergence in 1-2 iterations where GPU brute-force requires 1,800+ iterations."

### What AADC Is (For Every Summary/Abstract)

Every publication summary must include:

> "AADC (Adjoint Algorithmic Differentiation Compiler) is a C++ automatic differentiation library by MatLogica. Banks apply AADC to their existing pricing and risk code via operator overloading. This benchmark applies AADC to an ISDA SIMM implementation to measure performance across trading workflows."

### EOD Optimisation Story

When thousands of SIMM evaluations are needed to optimise trade allocation, GPU brute-force requires **18× more iterations** to converge than AADC. AADC provides exact gradients via automatic differentiation, converging in 1-2 iterations. GPU brute-force with default max_iters=100 doesn't converge at scale; with uncapped iterations it converges at ~1,848 iterations but takes **370× longer** than AADC.

### Workflow-Specific Framing

Don't lead with "AADC is 47× faster." Lead with "Pre-trade counterparty routing: AADC enables exploring 50+ counterparties in **under 2ms**; GPU brute-force requires **638ms** for 20 counterparties." Make speed advantage concrete in trading workflows.

### GPU Pathwise vs GPU Brute-Force

The benchmark includes two GPU approaches. gpu_full (pathwise) provides analytic gradients and converges like AADC — just 47× slower. bf_gpu (brute-force) cannot provide analytic gradients and requires dramatically more iterations to converge. Do not conflate these: the 1,047× headline applies to brute-force only. The 47× figure is the comparison against GPU's best approach.

### Normalise All Comparisons

Show per-evaluation latency for pre-trade checks (not raw wall-clock when eval counts differ). Show eval count for EOD optimisation (not just total time). Make clear bf_gpu's convergence requires 1,848 evals vs AADC's 9,114 evals but takes 370× longer wall-clock time.

---

## Pending Items Before Publication

1. ~~Explain cpp_aadc zero-iteration convergence mechanism~~ — **RESOLVED:** Was use-after-move bug, now reports 1-2 iterations
2. ~~Run bf_gpu with uncapped iterations~~ — **DONE:** Converges at 1,848 iterations, 370× slower than AADC
3. ~~Add continuous intraday monitoring workflow~~ — **RESOLVED:** What-If Step 4 already measures this
4. ~~Normalise all pre-trade comparisons to per-evaluation basis~~ — **DONE:** Updated Section 1.1
5. Document GPU optimisation attempts and profiling — occupancy, bandwidth analysis to prevent strawman criticism
6. Decide publication positioning: cpp_aadc vs aadc_full as headline — both should appear in every table; decide which goes in abstract/title
7. Test at larger portfolio sizes (10K, 100K trades) to demonstrate scaling
8. Add explicit AADC product description to every document summary/abstract
9. Ensure infographic/visual materials describe AADC as a differentiation library, not a SIMM calculator
10. Separate gpu_full (47×) and bf_gpu (1,047×) ratios clearly — don't let a single headline conflate both GPU approaches
11. **Add memory requirements section** — CPU/GPU memory by configuration
12. **Add methodology section** — SIMM formula, evaluation definition, trade generation

---

## Changes from Previous Draft

### Framing Corrections (New in This Revision)

- Removed "SIMM is a sensitivity-aggregation problem" as primary message — replaced with workflow-centric framing
- Added explicit section explaining what AADC is (differentiation library, not SIMM product)
- Added explicit section explaining what each backend is (computation strategies, not competing products)
- Added six specific misreading corrections (eval counts, convergence iterations, CPU requirements, IM consistency qualification, Python vs C++ labelling, GPU approach distinction)
- Added guidance to distinguish gpu_full (47×) from bf_gpu (1,047×) in all materials
- Added new pending items: AADC product description in every summary, infographic corrections, GPU approach separation

### Data Corrections (Based on Benchmark Investigation)

- **cpp_aadc iterations:** Changed from "0 iters" to "1-2 iters" (use-after-move bug fixed)
- **Pre-trade latency:** Updated to use pure kernel re-eval time (0.029ms) not workflow time
- **Counterparty routing:** Updated from 3.2ms to 0.58ms for 20 counterparties
- **bf_gpu convergence:** Added data showing it CAN converge with 1,848 iterations (~100s)
- **What-if ratio:** Corrected to 47× (34K vs 721) based on actual throughput data

### Structural Changes (Carried from First Amendment)

- Trading Impact ordered by commercial importance, not raw speedup
- EOD convergence story updated with uncapped iteration data
- Pre-trade checks lead with counterparty routing business case using pure kernel times
- Implementation section leads with convergence quality
- Removed cloud cost comparison (deferred pending research)
- De-emphasised portfolio setup and margin attribution (wall-clock differences not meaningful)
