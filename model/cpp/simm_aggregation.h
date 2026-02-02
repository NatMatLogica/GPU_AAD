// SIMM Aggregation Kernel with AADC Recording
// Ported from model/simm_portfolio_aadc_v2.py record_single_portfolio_simm_kernel_v2_full
// Version: 1.0.1
#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <iostream>
#include <chrono>
#include <memory>

#include <aadc/aadc.h>

#include "factor_metadata.h"
#include "simm_params_v26.h"

namespace simm {

using mmType = __m256d;
constexpr int AVX_WIDTH = sizeof(mmType) / sizeof(double);

// ============================================================================
// SIMM Kernel Structure
// ============================================================================
struct SIMMKernel {
    aadc::AADCFunctions<mmType> funcs;
    std::vector<aadc::AADCArgument> sens_handles;  // K handles
    aadc::AADCResult im_output;
    int K = 0;
    double recording_time_sec = 0.0;
};

// ============================================================================
// Internal: Compute margin for one risk class + one risk measure
// Called during AADC kernel recording with idouble types
// ============================================================================
inline idouble computeRiskClassMargin(
    RiskClass rc,
    const std::vector<int>& factor_indices,
    const std::vector<idouble>& agg_sens,
    const std::vector<FactorMeta>& metadata)
{
    if (factor_indices.empty()) {
        return idouble(0.0);
    }

    // Group factors by bucket_key
    std::unordered_map<std::string, std::vector<int>> buckets;
    for (int k : factor_indices) {
        buckets[metadata[k].bucket_key].push_back(k);
    }

    // Per-bucket: compute K_b and S_b
    struct BucketResult {
        idouble K_b;
        idouble S_b;
        double cr;
        int bucket_num;
    };
    std::vector<std::string> bucket_keys;
    std::unordered_map<std::string, BucketResult> bucket_results;

    for (auto& [bkey, bindices] : buckets) {
        bucket_keys.push_back(bkey);

        // Compute weighted sensitivities: WS_k = S_k * RW_k * CR_k
        std::vector<idouble> ws_list;
        ws_list.reserve(bindices.size());
        for (int k : bindices) {
            idouble ws_k = agg_sens[k] * (metadata[k].weight * metadata[k].cr);
            ws_list.push_back(ws_k);
        }

        double cr = metadata[bindices[0]].cr;
        int bucket_num = metadata[bindices[0]].bucket;

        // Compute K_b^2 with intra-bucket correlations
        idouble k_sq(0.0);
        idouble ws_sum(0.0);
        int n = static_cast<int>(bindices.size());

        if (n > 1) {
            for (int i = 0; i < n; ++i) {
                ws_sum = ws_sum + ws_list[i];
                for (int j = 0; j < n; ++j) {
                    double rho_ij = getIntraCorrelation(
                        rc, metadata[bindices[i]], metadata[bindices[j]]);
                    k_sq = k_sq + rho_ij * ws_list[i] * ws_list[j];
                }
            }
        } else {
            k_sq = ws_list[0] * ws_list[0];
            ws_sum = ws_list[0];
        }

        // Add tiny epsilon inside sqrt to avoid NaN derivative at zero
        bucket_results[bkey] = {sqrt(k_sq + idouble(1e-30)), ws_sum, cr, bucket_num};
    }

    // Inter-bucket aggregation
    idouble k_rc_sq(0.0);

    // Sum of K_b^2
    for (auto& bkey : bucket_keys) {
        auto& br = bucket_results[bkey];
        k_rc_sq = k_rc_sq + br.K_b * br.K_b;
    }

    // Cross-bucket gamma
    if (bucket_keys.size() > 1) {
        if (rc == RiskClass::Rates) {
            double gamma = v26::IR_GAMMA_DIFF_CCY;
            for (size_t i = 0; i < bucket_keys.size(); ++i) {
                for (size_t j = 0; j < bucket_keys.size(); ++j) {
                    if (i == j) continue;
                    auto& br_i = bucket_results[bucket_keys[i]];
                    auto& br_j = bucket_results[bucket_keys[j]];
                    double g_bc = 1.0;
                    if (std::max(br_i.cr, br_j.cr) > 0) {
                        g_bc = std::min(br_i.cr, br_j.cr) / std::max(br_i.cr, br_j.cr);
                    }
                    k_rc_sq = k_rc_sq + (gamma * g_bc) * br_i.S_b * br_j.S_b;
                }
            }
        } else {
            for (size_t i = 0; i < bucket_keys.size(); ++i) {
                for (size_t j = 0; j < bucket_keys.size(); ++j) {
                    if (i == j) continue;
                    auto& br_i = bucket_results[bucket_keys[i]];
                    auto& br_j = bucket_results[bucket_keys[j]];
                    double gamma = getInterBucketGamma(rc, br_i.bucket_num, br_j.bucket_num);
                    if (gamma != 0.0) {
                        double g_bc = 1.0;
                        if (std::max(br_i.cr, br_j.cr) > 0) {
                            g_bc = std::min(br_i.cr, br_j.cr) / std::max(br_i.cr, br_j.cr);
                        }
                        k_rc_sq = k_rc_sq + (gamma * g_bc) * br_i.S_b * br_j.S_b;
                    }
                }
            }
        }
    }

    return sqrt(k_rc_sq + idouble(1e-30));
}

// ============================================================================
// Record SIMM Aggregation Kernel
// K idouble inputs (aggregated sensitivities) -> 1 idouble output (total IM)
// ============================================================================
inline void recordSIMMKernel(
    SIMMKernel& kernel,
    int K,
    const std::vector<FactorMeta>& metadata)
{
    kernel.K = K;

    auto t_start = std::chrono::high_resolution_clock::now();

    kernel.funcs.startRecording();

    // Mark K aggregated sensitivities as differentiable inputs
    std::vector<idouble> agg_sens(K);
    kernel.sens_handles.resize(K);
    for (int k = 0; k < K; ++k) {
        agg_sens[k] = idouble(0.0);
        kernel.sens_handles[k] = agg_sens[k].markAsInput();
    }

    // Compute Delta and Vega margins per risk class
    std::vector<idouble> risk_class_margins(NUM_RISK_CLASSES);

    for (int rc_idx = 0; rc_idx < NUM_RISK_CLASSES; ++rc_idx) {
        RiskClass rc = static_cast<RiskClass>(rc_idx);

        // Separate delta and vega factor indices
        std::vector<int> delta_indices, vega_indices;
        for (int k = 0; k < K; ++k) {
            if (metadata[k].risk_class != rc) continue;
            if (metadata[k].risk_measure == RiskMeasure::Delta)
                delta_indices.push_back(k);
            else if (metadata[k].risk_measure == RiskMeasure::Vega)
                vega_indices.push_back(k);
        }

        // Delta margin
        idouble delta_margin = computeRiskClassMargin(
            rc, delta_indices, agg_sens, metadata);

        // Vega margin
        idouble vega_margin = computeRiskClassMargin(
            rc, vega_indices, agg_sens, metadata);

        risk_class_margins[rc_idx] = delta_margin + vega_margin;
    }

    // Cross-risk-class aggregation: IM = sqrt(sum_r sum_s PSI[r,s] * K_r * K_s)
    idouble simm_sq(0.0);
    for (int i = 0; i < NUM_RISK_CLASSES; ++i) {
        for (int j = 0; j < NUM_RISK_CLASSES; ++j) {
            double psi_ij = v26::PSI[i][j];
            simm_sq = simm_sq + psi_ij * risk_class_margins[i] * risk_class_margins[j];
        }
    }

    idouble im = sqrt(simm_sq + idouble(1e-30));
    kernel.im_output = im.markAsOutput();

    kernel.funcs.stopRecording();

    auto t_end = std::chrono::high_resolution_clock::now();
    kernel.recording_time_sec = std::chrono::duration<double>(t_end - t_start).count();
}

} // namespace simm
