// SIMM Analytics: Margin Attribution, What-If Scenarios, Pre-Trade Routing
// Version: 1.1.0
#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <variant>

#include "simm_aggregation.h"
#include "sensitivity_matrix.h"
#include "allocation_optimizer.h"
#include "xccy_swap.h"

namespace simm {

using aadc::mmSetConst;
using aadc::mmSum;

// ============================================================================
// Trade Attribution Result
// ============================================================================
struct TradeAttribution {
    int trade_idx;
    std::string trade_id;
    double contribution;      // Euler-allocated IM contribution
    double pct_of_total;      // contribution / total_im * 100
};

struct AttributionResult {
    std::vector<TradeAttribution> attributions;  // sorted by |contribution| desc
    double total_im;
    double sum_contributions;  // should == total_im (Euler property)
    double eval_time_sec;
};

// ============================================================================
// What-If Scenario Result
// ============================================================================
struct WhatIfResult {
    double base_im;
    double scenario_im;
    double im_change;
    double im_change_pct;
    std::string description;
    std::vector<std::string> trades_affected;
    double eval_time_sec;
};

// ============================================================================
// Counterparty Routing Result
// ============================================================================
struct RoutingResult {
    std::vector<double> marginal_ims;   // P values
    std::vector<double> base_ims;       // P current IM values
    int best_portfolio;
    double best_marginal_im;
    double standalone_im;
    double eval_time_sec;
};

// ============================================================================
// Bilateral vs Cleared Comparison
// ============================================================================
struct ClearingComparison {
    double simm_margin;
    double ccp_margin;
    double margin_difference;    // simm - ccp
    double difference_pct;
    std::string recommendation;  // "BILATERAL", "CLEARED", "INDIFFERENT"
};

// ============================================================================
// Helper: Extract K-dimensional gradient from single-portfolio evaluation
// ============================================================================
inline std::vector<double> computePortfolioGradientK(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    double& out_im)
{
    int K = kernel.K;

    // Aggregate all trade sensitivities: agg_S[k] = sum_t S[t,k]
    std::vector<double> agg_S(K, 0.0);
    for (int t = 0; t < S.T; ++t) {
        for (int k = 0; k < K; ++k) {
            agg_S[k] += S(t, k);
        }
    }

    // Evaluate kernel once (single portfolio in lane 0)
    auto ws = kernel.funcs.createWorkSpace();
    for (int k = 0; k < K; ++k) {
        ws->setVal(kernel.sens_handles[k], mmSetConst<mmType>(agg_S[k]));
    }
    kernel.funcs.forward(*ws);

    // Extract IM
    {
        mmType mm_im = ws->val(kernel.im_output);
        out_im = reinterpret_cast<double*>(&mm_im)[0];
    }

    // Reverse pass
    ws->resetDiff();
    ws->setDiff(kernel.im_output, 1.0);
    kernel.funcs.reverse(*ws);

    // Extract K gradients (lane 0)
    std::vector<double> grad(K);
    for (int k = 0; k < K; ++k) {
        mmType mm_g = ws->diff(kernel.sens_handles[k]);
        grad[k] = reinterpret_cast<double*>(&mm_g)[0];
    }
    return grad;
}

// ============================================================================
// Helper: Forward-only single-portfolio IM from pre-aggregated agg_S[K]
// No matmul, no reverse pass — O(K) setup + forward AADC only.
// ============================================================================
inline double evaluateIMFromAggSSingle(SIMMKernel& kernel, const double* agg_S) {
    int K = kernel.K;
    auto ws = kernel.funcs.createWorkSpace();
    for (int k = 0; k < K; ++k) {
        ws->setVal(kernel.sens_handles[k], mmSetConst<mmType>(agg_S[k]));
    }
    kernel.funcs.forward(*ws);
    mmType mm_im = ws->val(kernel.im_output);
    return reinterpret_cast<double*>(&mm_im)[0];
}

// ============================================================================
// Helper: Forward+reverse single-portfolio IM + K-dimensional gradient
// from pre-aggregated agg_S[K]. No matmul needed.
// ============================================================================
inline double evaluateIMAndGradFromAggS(
    SIMMKernel& kernel, const double* agg_S, double* grad_out, int K)
{
    auto ws = kernel.funcs.createWorkSpace();
    for (int k = 0; k < K; ++k) {
        ws->setVal(kernel.sens_handles[k], mmSetConst<mmType>(agg_S[k]));
    }
    kernel.funcs.forward(*ws);
    mmType mm_im = ws->val(kernel.im_output);
    double im = reinterpret_cast<double*>(&mm_im)[0];

    ws->resetDiff();
    ws->setDiff(kernel.im_output, 1.0);
    kernel.funcs.reverse(*ws);

    for (int k = 0; k < K; ++k) {
        mmType mm_g = ws->diff(kernel.sens_handles[k]);
        grad_out[k] = reinterpret_cast<double*>(&mm_g)[0];
    }
    return im;
}

// ============================================================================
// Helper: Evaluate SIMM for a single portfolio given sensitivity matrix
// ============================================================================
inline double evaluateSinglePortfolioIM(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    int num_threads)
{
    AllocationMatrix alloc(S.T, 1);
    for (int t = 0; t < S.T; ++t) alloc(t, 0) = 1.0;
    auto eval = evaluateAllPortfolios(kernel, S, alloc, num_threads);
    return eval.ims[0];
}

// ============================================================================
// Margin Attribution: Euler Decomposition via AADC gradient
//
// contribution[t] = sum_k S[t,k] * dIM/dAggS_k
// This is exactly what evaluateAllPortfolios returns as gradient[t,0] for P=1
// ============================================================================
inline AttributionResult computeMarginAttribution(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    int num_threads,
    int top_n = 0)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    int T = S.T;

    // Single portfolio: all trades in one group
    AllocationMatrix alloc(T, 1);
    for (int t = 0; t < T; ++t) alloc(t, 0) = 1.0;

    // Evaluate: gradient[t] = sum_k S[t,k] * dIM/dAggS_k (chain rule)
    auto eval = evaluateAllPortfolios(kernel, S, alloc, num_threads);
    double total_im = eval.ims[0];

    // Build attribution list
    std::vector<TradeAttribution> attrs(T);
    double sum = 0.0;
    for (int t = 0; t < T; ++t) {
        attrs[t].trade_idx = t;
        attrs[t].trade_id = (t < static_cast<int>(S.trade_ids.size()))
            ? S.trade_ids[t] : ("T" + std::to_string(t));
        attrs[t].contribution = eval.gradient[t];  // T x 1, index = t
        attrs[t].pct_of_total = (total_im > 1e-15)
            ? 100.0 * attrs[t].contribution / total_im : 0.0;
        sum += attrs[t].contribution;
    }

    // Sort by |contribution| descending
    std::sort(attrs.begin(), attrs.end(),
              [](const TradeAttribution& a, const TradeAttribution& b) {
                  return std::abs(a.contribution) > std::abs(b.contribution);
              });

    if (top_n > 0 && top_n < T) attrs.resize(top_n);

    auto t_end = std::chrono::high_resolution_clock::now();

    return {std::move(attrs), total_im, sum,
            std::chrono::duration<double>(t_end - t_start).count()};
}

// ============================================================================
// What-If: Unwind Top N Contributors
// ============================================================================
inline WhatIfResult whatIfUnwindTopN(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    int num_threads,
    int n_unwind)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    // Get attribution to find top N
    auto attr = computeMarginAttribution(kernel, S, num_threads);
    double base_im = attr.total_im;

    // Build modified S with top N trades zeroed out
    SensitivityMatrix S_mod;
    S_mod.T = S.T;
    S_mod.K = S.K;
    S_mod.data = S.data;  // copy
    S_mod.risk_factors = S.risk_factors;
    S_mod.trade_ids = S.trade_ids;

    int actual_unwind = std::min(n_unwind, static_cast<int>(attr.attributions.size()));
    std::vector<std::string> affected;
    for (int i = 0; i < actual_unwind; ++i) {
        int tidx = attr.attributions[i].trade_idx;
        for (int k = 0; k < S.K; ++k) {
            S_mod.data[tidx * S.K + k] = 0.0;
        }
        affected.push_back(attr.attributions[i].trade_id);
    }

    // Re-evaluate SIMM with modified S
    double scenario_im = evaluateSinglePortfolioIM(kernel, S_mod, num_threads);

    auto t_end = std::chrono::high_resolution_clock::now();

    double change = scenario_im - base_im;
    double change_pct = (base_im > 1e-15) ? 100.0 * change / base_im : 0.0;

    return {base_im, scenario_im, change, change_pct,
            "Unwind top " + std::to_string(actual_unwind) + " contributors",
            std::move(affected),
            std::chrono::duration<double>(t_end - t_start).count()};
}

// ============================================================================
// What-If: Add Hedge Trade
// ============================================================================
inline WhatIfResult whatIfAddHedge(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    const std::vector<CRIFRecord>& hedge_crif,
    int num_threads,
    const std::string& hedge_desc = "hedge")
{
    auto t_start = std::chrono::high_resolution_clock::now();

    double base_im = evaluateSinglePortfolioIM(kernel, S, num_threads);

    // Build risk factor key -> column index map
    std::unordered_map<std::string, int> rf_index;
    for (int k = 0; k < S.K; ++k) {
        auto& [rt, q, b, l] = S.risk_factors[k];
        rf_index[rt + "|" + q + "|" + std::to_string(b) + "|" + l] = k;
    }

    // Build extended S with hedge appended as row T
    SensitivityMatrix S_ext;
    S_ext.T = S.T + 1;
    S_ext.K = S.K;
    S_ext.data.resize(S_ext.T * S_ext.K, 0.0);
    std::copy(S.data.begin(), S.data.end(), S_ext.data.begin());
    S_ext.risk_factors = S.risk_factors;
    S_ext.trade_ids = S.trade_ids;
    S_ext.trade_ids.push_back(hedge_desc);

    // Map hedge CRIF to columns
    for (auto& rec : hedge_crif) {
        std::string key = rec.risk_type + "|" + rec.qualifier + "|" +
                          std::to_string(rec.bucket) + "|" + rec.label1;
        auto it = rf_index.find(key);
        if (it != rf_index.end()) {
            S_ext.data[S.T * S.K + it->second] += rec.amount;
        }
    }

    double scenario_im = evaluateSinglePortfolioIM(kernel, S_ext, num_threads);

    auto t_end = std::chrono::high_resolution_clock::now();

    double change = scenario_im - base_im;
    double change_pct = (base_im > 1e-15) ? 100.0 * change / base_im : 0.0;

    return {base_im, scenario_im, change, change_pct,
            "Add " + hedge_desc,
            {hedge_desc},
            std::chrono::duration<double>(t_end - t_start).count()};
}

// ============================================================================
// What-If: Stress Scenario
// ============================================================================
inline WhatIfResult whatIfStressScenario(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    double shock_factor,
    int num_threads,
    const std::vector<FactorMeta>& metadata = {},
    RiskClass target_rc = RiskClass::Rates)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    double base_im = evaluateSinglePortfolioIM(kernel, S, num_threads);

    // Copy and apply shock
    SensitivityMatrix S_stress;
    S_stress.T = S.T;
    S_stress.K = S.K;
    S_stress.data = S.data;
    S_stress.risk_factors = S.risk_factors;
    S_stress.trade_ids = S.trade_ids;

    if (!metadata.empty()) {
        // Shock only factors matching target risk class
        for (int k = 0; k < S.K; ++k) {
            if (metadata[k].risk_class == target_rc) {
                for (int t = 0; t < S.T; ++t) {
                    S_stress.data[t * S.K + k] *= shock_factor;
                }
            }
        }
    } else {
        // Shock all factors
        for (auto& v : S_stress.data) v *= shock_factor;
    }

    double scenario_im = evaluateSinglePortfolioIM(kernel, S_stress, num_threads);

    auto t_end = std::chrono::high_resolution_clock::now();

    double change = scenario_im - base_im;
    double change_pct = (base_im > 1e-15) ? 100.0 * change / base_im : 0.0;

    std::string rc_name = metadata.empty() ? "All"
        : std::string(riskClassName(target_rc));

    return {base_im, scenario_im, change, change_pct,
            rc_name + " stress x" + std::to_string(shock_factor),
            {},
            std::chrono::duration<double>(t_end - t_start).count()};
}

// ============================================================================
// Marginal IM: O(K) dot product using pre-computed gradient
// ============================================================================
inline double computeMarginalIM(
    const std::vector<double>& gradient_K,
    const std::vector<double>& new_trade_sens,
    int K)
{
    double result = 0.0;
    for (int k = 0; k < K; ++k) {
        result += gradient_K[k] * new_trade_sens[k];
    }
    return result;
}

// ============================================================================
// Counterparty Routing: Find best portfolio for a new trade
// ============================================================================
inline RoutingResult counterpartyRouting(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    const AllocationMatrix& current_alloc,
    const std::vector<CRIFRecord>& new_trade_crif,
    int num_threads)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    int P = current_alloc.P;
    int K = kernel.K;

    // Step 1: Evaluate all P portfolios with gradients
    auto eval = evaluateAllPortfoliosMT(kernel, S, current_alloc, num_threads);

    // Step 2: Extract per-portfolio K-dimensional gradients
    // Need to re-evaluate to get K-dimensional gradients (not T-dimensional)
    std::vector<double> agg_S(K * P);
    matmulATB(S.data.data(), current_alloc.data.data(), agg_S.data(), S.T, K, P);

    auto ws = kernel.funcs.createWorkSpace();
    int num_batches = (P + AVX_WIDTH - 1) / AVX_WIDTH;

    std::vector<std::vector<double>> grad_per_portfolio(P, std::vector<double>(K));

    for (int batch = 0; batch < num_batches; ++batch) {
        int p_start = batch * AVX_WIDTH;

        for (int k = 0; k < K; ++k) {
            mmType mm_val;
            double* ptr = reinterpret_cast<double*>(&mm_val);
            for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                int p = p_start + lane;
                // Padding lanes copy lane 0 to avoid sqrt(0)->NaN in derivative
                ptr[lane] = (p < P) ? agg_S[k * P + p] : agg_S[k * P + p_start];
            }
            ws->setVal(kernel.sens_handles[k], mm_val);
        }
        kernel.funcs.forward(*ws);
        ws->resetDiff();
        ws->setDiff(kernel.im_output, 1.0);
        kernel.funcs.reverse(*ws);

        for (int k = 0; k < K; ++k) {
            mmType mm_g = ws->diff(kernel.sens_handles[k]);
            double* gp = reinterpret_cast<double*>(&mm_g);
            for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                int p = p_start + lane;
                if (p < P) grad_per_portfolio[p][k] = gp[lane];
            }
        }
    }

    // Step 3: Map new trade CRIF to K-vector
    std::vector<double> new_sens(K, 0.0);
    std::unordered_map<std::string, int> rf_index;
    for (int k = 0; k < K; ++k) {
        auto& [rt, q, b, l] = S.risk_factors[k];
        rf_index[rt + "|" + q + "|" + std::to_string(b) + "|" + l] = k;
    }
    for (auto& rec : new_trade_crif) {
        std::string key = rec.risk_type + "|" + rec.qualifier + "|" +
                          std::to_string(rec.bucket) + "|" + rec.label1;
        auto it = rf_index.find(key);
        if (it != rf_index.end()) new_sens[it->second] = rec.amount;
    }

    // Step 4: Compute standalone IM for new trade
    double standalone_im = 0.0;
    {
        // Single-factor approximation: sum of |sens| * risk_weight
        for (int k = 0; k < K; ++k) {
            standalone_im += std::abs(new_sens[k]);
        }
    }

    // Step 5: Marginal IM per portfolio
    std::vector<double> marginals(P);
    int best_p = 0;
    double best_mim = std::numeric_limits<double>::max();
    for (int p = 0; p < P; ++p) {
        marginals[p] = computeMarginalIM(grad_per_portfolio[p], new_sens, K);
        if (marginals[p] < best_mim) {
            best_mim = marginals[p];
            best_p = p;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    return {std::move(marginals), std::move(eval.ims), best_p, best_mim,
            standalone_im,
            std::chrono::duration<double>(t_end - t_start).count()};
}

// ============================================================================
// Bilateral vs Cleared Comparison
// ============================================================================
template <typename Trade>
inline ClearingComparison bilateralVsCleared(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    const std::vector<Trade>& portfolio,
    int num_threads,
    double ccp_margin_rate = 0.03)
{
    // SIMM margin (full kernel evaluation)
    double simm = evaluateSinglePortfolioIM(kernel, S, num_threads);

    // CCP margin: simplified as fraction of gross notional
    double gross_notional = 0.0;
    for (auto& trade : portfolio) {
        std::visit([&](const auto& t) {
            using TT = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<TT, XCCYSwapTrade>)
                gross_notional += std::abs(t.dom_notional);
            else
                gross_notional += std::abs(t.notional);
        }, trade);
    }
    double ccp = gross_notional * ccp_margin_rate;

    double diff = simm - ccp;
    double pct = (simm > 1e-15) ? 100.0 * diff / simm : 0.0;

    std::string rec;
    if (pct > 10.0) rec = "CLEARED";
    else if (pct < -5.0) rec = "BILATERAL";
    else rec = "INDIFFERENT";

    return {simm, ccp, diff, pct, rec};
}

} // namespace simm
