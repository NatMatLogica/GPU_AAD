// Sensitivity Matrix: T trades x K risk factors
// Version: 1.0.0
#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include "factor_metadata.h"

namespace simm {

// ============================================================================
// Sensitivity Matrix (row-major T x K)
// ============================================================================
struct SensitivityMatrix {
    std::vector<double> data;  // Row-major T x K
    int T;                      // Number of trades
    int K;                      // Number of unique risk factors
    std::vector<RiskFactorKey> risk_factors;  // K factor descriptors
    std::vector<std::string> trade_ids;       // T trade IDs

    double& operator()(int t, int k) { return data[t * K + k]; }
    double operator()(int t, int k) const { return data[t * K + k]; }

    // Column access for aggregation: returns pointer to S[0, k]
    // Stride is K (row-major)
    const double* colPtr(int k) const { return &data[k]; }
};

// ============================================================================
// CRIF-like record for a single sensitivity
// ============================================================================
struct CRIFRecord {
    std::string risk_type;
    std::string qualifier;
    int bucket;
    std::string label1;
    double amount;
};

// ============================================================================
// Build sensitivity matrix from per-trade CRIFs
// ============================================================================
inline SensitivityMatrix buildSensitivityMatrix(
    const std::vector<std::string>& trade_ids,
    const std::vector<std::vector<CRIFRecord>>& trade_crifs)
{
    int T = static_cast<int>(trade_crifs.size());
    assert(T == static_cast<int>(trade_ids.size()));

    // Collect all unique risk factor keys
    // Use a stable ordering via vector + set
    std::vector<RiskFactorKey> rf_keys;
    std::unordered_map<std::string, int> rf_index; // string key -> column index

    auto makeKey = [](const CRIFRecord& r) -> std::string {
        return r.risk_type + "|" + r.qualifier + "|" +
               std::to_string(r.bucket) + "|" + r.label1;
    };

    for (const auto& crif : trade_crifs) {
        for (const auto& rec : crif) {
            std::string skey = makeKey(rec);
            if (rf_index.find(skey) == rf_index.end()) {
                int idx = static_cast<int>(rf_keys.size());
                rf_index[skey] = idx;
                rf_keys.emplace_back(rec.risk_type, rec.qualifier, rec.bucket, rec.label1);
            }
        }
    }

    int K = static_cast<int>(rf_keys.size());

    // Build matrix
    SensitivityMatrix sm;
    sm.T = T;
    sm.K = K;
    sm.risk_factors = std::move(rf_keys);
    sm.trade_ids = trade_ids;
    sm.data.assign(T * K, 0.0);

    for (int t = 0; t < T; ++t) {
        for (const auto& rec : trade_crifs[t]) {
            std::string skey = makeKey(rec);
            int k = rf_index[skey];
            sm(t, k) += rec.amount;
        }
    }

    return sm;
}

// ============================================================================
// Compute bucket-level sum of sensitivities (for concentration factors)
// Returns map of "RiskClass|BucketKey" -> sum of absolute sensitivities
// ============================================================================
inline std::unordered_map<std::string, double> computeBucketSumSensitivities(
    const SensitivityMatrix& sm,
    const std::vector<RiskFactorKey>& risk_factors)
{
    std::unordered_map<std::string, double> result;

    for (int k = 0; k < sm.K; ++k) {
        auto& [rt, qualifier, bucket, label1] = risk_factors[k];
        RiskClass rc = mapRiskTypeToClass(rt);
        std::string bucket_key;
        if (rc == RiskClass::Rates || rc == RiskClass::FX)
            bucket_key = qualifier;
        else
            bucket_key = std::to_string(bucket);

        std::string key = std::string(riskClassName(rc)) + "|" + bucket_key;

        double col_sum = 0.0;
        for (int t = 0; t < sm.T; ++t) {
            col_sum += sm(t, k);
        }
        result[key] += col_sum;
    }

    return result;
}

} // namespace simm
