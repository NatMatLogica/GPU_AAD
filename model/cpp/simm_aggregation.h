// SIMM Aggregation Kernel with AADC Recording
// Ported from model/simm_portfolio_aadc_v2.py record_single_portfolio_simm_kernel_v2_full
// Version: 2.0.0
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

// ============================================================================
// Pre-Weighted SIMM Kernel: inputs are WS = s * rw * cr (pre-computed)
// Reduces tape size by moving weight/CR multiplication outside kernel.
// Gradients are dIM/dWS; caller applies chain rule: dIM/ds = dIM/dWS * rw * cr
// ============================================================================

inline idouble computeRiskClassMarginPreWeighted(
    RiskClass rc,
    const std::vector<int>& factor_indices,
    const std::vector<idouble>& ws_inputs,
    const std::vector<FactorMeta>& metadata,
    bool use_correlations)
{
    if (factor_indices.empty()) {
        return idouble(0.0);
    }

    // Group factors by bucket_key
    std::unordered_map<std::string, std::vector<int>> buckets;
    for (int k : factor_indices) {
        buckets[metadata[k].bucket_key].push_back(k);
    }

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

        idouble k_sq(0.0);
        idouble ws_sum(0.0);
        int n = static_cast<int>(bindices.size());

        if (use_correlations && n > 1) {
            for (int i = 0; i < n; ++i) {
                ws_sum = ws_sum + ws_inputs[bindices[i]];
                for (int j = 0; j < n; ++j) {
                    double rho_ij = getIntraCorrelation(
                        rc, metadata[bindices[i]], metadata[bindices[j]]);
                    k_sq = k_sq + rho_ij * ws_inputs[bindices[i]] * ws_inputs[bindices[j]];
                }
            }
        } else if (n > 1) {
            // Diagonal only: sum of squares
            for (int i = 0; i < n; ++i) {
                k_sq = k_sq + ws_inputs[bindices[i]] * ws_inputs[bindices[i]];
                ws_sum = ws_sum + ws_inputs[bindices[i]];
            }
        } else {
            k_sq = ws_inputs[bindices[0]] * ws_inputs[bindices[0]];
            ws_sum = ws_inputs[bindices[0]];
        }

        double cr = metadata[bindices[0]].cr;
        int bucket_num = metadata[bindices[0]].bucket;
        bucket_results[bkey] = {sqrt(k_sq + idouble(1e-30)), ws_sum, cr, bucket_num};
    }

    // Inter-bucket aggregation (same logic as standard kernel)
    idouble k_rc_sq(0.0);
    for (auto& bkey : bucket_keys) {
        auto& br = bucket_results[bkey];
        k_rc_sq = k_rc_sq + br.K_b * br.K_b;
    }

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

inline void recordSIMMKernelPreWeighted(
    SIMMKernel& kernel,
    int K,
    const std::vector<FactorMeta>& metadata,
    bool use_correlations = true)
{
    kernel.K = K;

    auto t_start = std::chrono::high_resolution_clock::now();

    kernel.funcs.startRecording();

    // Inputs are pre-weighted sensitivities: WS_k = s_k * rw_k * cr_k
    std::vector<idouble> ws(K);
    kernel.sens_handles.resize(K);
    for (int k = 0; k < K; ++k) {
        ws[k] = idouble(0.0);
        kernel.sens_handles[k] = ws[k].markAsInput();
    }

    // Per risk class margin (no weight/CR multiplication -- already applied)
    std::vector<idouble> risk_class_margins(NUM_RISK_CLASSES);

    for (int rc_idx = 0; rc_idx < NUM_RISK_CLASSES; ++rc_idx) {
        RiskClass rc = static_cast<RiskClass>(rc_idx);

        std::vector<int> delta_indices, vega_indices;
        for (int k = 0; k < K; ++k) {
            if (metadata[k].risk_class != rc) continue;
            if (metadata[k].risk_measure == RiskMeasure::Delta)
                delta_indices.push_back(k);
            else if (metadata[k].risk_measure == RiskMeasure::Vega)
                vega_indices.push_back(k);
        }

        idouble delta_margin = computeRiskClassMarginPreWeighted(
            rc, delta_indices, ws, metadata, use_correlations);
        idouble vega_margin = computeRiskClassMarginPreWeighted(
            rc, vega_indices, ws, metadata, use_correlations);

        risk_class_margins[rc_idx] = delta_margin + vega_margin;
    }

    // Cross-risk-class aggregation
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

// ============================================================================
// SIMM Kernel Cache
// Avoids re-recording when factor structure (K + risk_type/qualifier/bucket/label1)
// is unchanged across calls.
// ============================================================================

struct SIMMKernelCacheKey {
    int K;
    size_t factor_hash;
    bool pre_weighted;

    bool operator==(const SIMMKernelCacheKey& other) const {
        return K == other.K && factor_hash == other.factor_hash
            && pre_weighted == other.pre_weighted;
    }
};

struct SIMMKernelCacheKeyHash {
    size_t operator()(const SIMMKernelCacheKey& key) const {
        size_t h = std::hash<int>()(key.K);
        h ^= key.factor_hash + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(key.pre_weighted) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class SIMMKernelCache {
public:
    static SIMMKernelCacheKey makeKey(
        int K, const std::vector<FactorMeta>& metadata, bool pre_weighted = false)
    {
        size_t h = 0;
        for (int k = 0; k < K; ++k) {
            h ^= std::hash<std::string>()(metadata[k].risk_type) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>()(metadata[k].qualifier) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(metadata[k].bucket) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>()(metadata[k].label1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return {K, h, pre_weighted};
    }

    SIMMKernel* get(const SIMMKernelCacheKey& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            ++hits_;
            return it->second.get();
        }
        ++misses_;
        return nullptr;
    }

    SIMMKernel& put(const SIMMKernelCacheKey& key, std::unique_ptr<SIMMKernel> kernel) {
        auto& ref = cache_[key];
        ref = std::move(kernel);
        return *ref;
    }

    int hits() const { return hits_; }
    int misses() const { return misses_; }
    int size() const { return static_cast<int>(cache_.size()); }

private:
    std::unordered_map<SIMMKernelCacheKey, std::unique_ptr<SIMMKernel>,
                       SIMMKernelCacheKeyHash> cache_;
    int hits_ = 0;
    int misses_ = 0;
};

inline SIMMKernel& getOrRecordSIMMKernel(
    SIMMKernelCache& cache,
    int K,
    const std::vector<FactorMeta>& metadata,
    bool pre_weighted = false,
    bool use_correlations = true)
{
    auto key = SIMMKernelCache::makeKey(K, metadata, pre_weighted);
    SIMMKernel* cached = cache.get(key);
    if (cached) return *cached;

    auto kernel = std::make_unique<SIMMKernel>();
    if (pre_weighted) {
        recordSIMMKernelPreWeighted(*kernel, K, metadata, use_correlations);
    } else {
        recordSIMMKernel(*kernel, K, metadata);
    }
    return cache.put(key, std::move(kernel));
}

} // namespace simm
