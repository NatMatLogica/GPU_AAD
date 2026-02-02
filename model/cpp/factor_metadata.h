// Factor Metadata for SIMM AADC Kernel
// Ported from model/simm_portfolio_aadc_v2.py FactorMeta
// Version: 1.0.0
#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include "simm_params_v26.h"

namespace simm {

// ============================================================================
// Risk Class and Measure Enums
// ============================================================================
enum class RiskClass { Rates=0, CreditQ=1, CreditNonQ=2, Equity=3, Commodity=4, FX=5 };
enum class RiskMeasure { Delta, Vega };
constexpr int NUM_RISK_CLASSES = 6;

inline const char* riskClassName(RiskClass rc) {
    switch (rc) {
        case RiskClass::Rates:      return "Rates";
        case RiskClass::CreditQ:    return "CreditQ";
        case RiskClass::CreditNonQ: return "CreditNonQ";
        case RiskClass::Equity:     return "Equity";
        case RiskClass::Commodity:  return "Commodity";
        case RiskClass::FX:         return "FX";
    }
    return "Unknown";
}

// ============================================================================
// Factor Metadata
// ============================================================================
struct FactorMeta {
    RiskClass risk_class;
    RiskMeasure risk_measure;
    std::string risk_type;    // "Risk_IRCurve", "Risk_FX", etc.
    std::string qualifier;    // Currency, issuer
    int bucket;               // Bucket number (0 = residual)
    std::string label1;       // Tenor label for IR
    int tenor_idx;            // Pre-computed tenor index (0-11 for IR, -1 otherwise)
    double weight;            // RW (Delta) or VRW (Vega)
    double cr;                // Concentration risk factor
    std::string bucket_key;   // Grouping key (currency for IR/FX, bucket string for others)
};

// ============================================================================
// Risk Type Classification
// ============================================================================
inline bool isDeltaRiskType(const std::string& rt) {
    return rt == "Risk_IRCurve" || rt == "Risk_Inflation" || rt == "Risk_XCcyBasis" ||
           rt == "Risk_FX" || rt == "Risk_Equity" || rt == "Risk_CreditQ" ||
           rt == "Risk_CreditNonQ" || rt == "Risk_Commodity";
}

inline bool isVegaRiskType(const std::string& rt) {
    return rt == "Risk_IRVol" || rt == "Risk_InflationVol" ||
           rt == "Risk_FXVol" || rt == "Risk_EquityVol" ||
           rt == "Risk_CreditVol" || rt == "Risk_CreditVolNonQ" ||
           rt == "Risk_CommodityVol";
}

inline RiskClass mapRiskTypeToClass(const std::string& rt) {
    if (rt == "Risk_IRCurve" || rt == "Risk_Inflation" || rt == "Risk_XCcyBasis" ||
        rt == "Risk_IRVol" || rt == "Risk_InflationVol")
        return RiskClass::Rates;
    if (rt == "Risk_CreditQ" || rt == "Risk_CreditVol" || rt == "Risk_BaseCorr")
        return RiskClass::CreditQ;
    if (rt == "Risk_CreditNonQ" || rt == "Risk_CreditVolNonQ")
        return RiskClass::CreditNonQ;
    if (rt == "Risk_Equity" || rt == "Risk_EquityVol")
        return RiskClass::Equity;
    if (rt == "Risk_Commodity" || rt == "Risk_CommodityVol")
        return RiskClass::Commodity;
    if (rt == "Risk_FX" || rt == "Risk_FXVol")
        return RiskClass::FX;
    return RiskClass::Rates; // default
}

// ============================================================================
// Risk Weight Lookups
// ============================================================================
inline double getRiskWeight(const std::string& rt, const std::string& qualifier,
                            int bucket, const std::string& label1) {
    RiskClass rc = mapRiskTypeToClass(rt);

    if (rt == "Risk_IRCurve")
        return v26::getIRRiskWeight(qualifier, label1);
    if (rt == "Risk_Inflation")
        return v26::INFLATION_RW;
    if (rt == "Risk_XCcyBasis")
        return v26::CCY_BASIS_SWAP_SPREAD_RW;

    if (rc == RiskClass::CreditQ) {
        auto it = v26::CREDITQ_RW.find(bucket);
        return (it != v26::CREDITQ_RW.end()) ? it->second : 343.0;
    }
    if (rc == RiskClass::CreditNonQ) {
        auto it = v26::CREDITNONQ_RW.find(bucket);
        return (it != v26::CREDITNONQ_RW.end()) ? it->second : 1300.0;
    }
    if (rc == RiskClass::Equity) {
        auto it = v26::EQUITY_RW.find(bucket);
        return (it != v26::EQUITY_RW.end()) ? it->second : 50.0;
    }
    if (rc == RiskClass::Commodity) {
        auto it = v26::COMMODITY_RW.find(bucket);
        return (it != v26::COMMODITY_RW.end()) ? it->second : 68.0;
    }
    if (rc == RiskClass::FX)
        return v26::getFXRiskWeight("USD", qualifier); // default calc ccy

    return 1.0;
}

inline double getVegaRiskWeight(const std::string& rt, int bucket) {
    RiskClass rc = mapRiskTypeToClass(rt);
    switch (rc) {
        case RiskClass::Rates:      return v26::IR_VRW;
        case RiskClass::CreditQ:    return v26::CREDITQ_VRW;
        case RiskClass::CreditNonQ: return v26::CREDITNONQ_VRW;
        case RiskClass::Equity:
            return (bucket == 12) ? v26::EQUITY_VRW_BUCKET_12 : v26::EQUITY_VRW;
        case RiskClass::Commodity:  return v26::COMMODITY_VRW;
        case RiskClass::FX:         return v26::FX_VRW;
    }
    return 1.0;
}

// ============================================================================
// Intra-Bucket Correlation
// ============================================================================
inline double getIntraCorrelation(RiskClass rc, const FactorMeta& f1, const FactorMeta& f2,
                                   const std::string& calcCcy = "USD") {
    if (rc == RiskClass::Rates) {
        // XCcyBasis
        if (f1.risk_type == "Risk_XCcyBasis" || f2.risk_type == "Risk_XCcyBasis") {
            if (f1.risk_type == f2.risk_type) return 1.0;
            return v26::CCY_BASIS_SPREAD_CORR;
        }
        // Inflation
        if (f1.risk_type == "Risk_Inflation" || f2.risk_type == "Risk_Inflation" ||
            f1.risk_type == "Risk_InflationVol" || f2.risk_type == "Risk_InflationVol") {
            if (f1.risk_type == f2.risk_type) return 1.0;
            return v26::INFLATION_CORR;
        }
        // IR curve: tenor correlation
        if (f1.tenor_idx >= 0 && f2.tenor_idx >= 0) {
            return v26::IR_CORR[f1.tenor_idx][f2.tenor_idx];
        }
        return 1.0;
    }

    if (rc == RiskClass::Equity) {
        auto it = v26::EQUITY_CORR.find(f1.bucket);
        return (it != v26::EQUITY_CORR.end()) ? it->second : 0.25;
    }

    if (rc == RiskClass::Commodity) {
        auto it = v26::COMMODITY_CORR.find(f1.bucket);
        return (it != v26::COMMODITY_CORR.end()) ? it->second : 0.5;
    }

    if (rc == RiskClass::FX) {
        return v26::getFXDeltaCorrelation(calcCcy, f1.qualifier, f2.qualifier);
    }

    if (rc == RiskClass::CreditQ) {
        // Simplified: use bucket-level correlation
        static const std::unordered_map<int, double> creditQCorrByBucket = {
            {1,0.73},{2,0.69},{3,0.69},{4,0.67},{5,0.67},{6,0.67},
            {7,0.73},{8,0.65},{9,0.62},{10,0.61},{11,0.48},{12,0.38},{0,0.50}
        };
        auto it = creditQCorrByBucket.find(f1.bucket);
        return (it != creditQCorrByBucket.end()) ? it->second : 0.50;
    }

    if (rc == RiskClass::CreditNonQ) {
        static const std::unordered_map<int, double> creditNonQCorr = {
            {1,0.43},{2,0.77},{0,0.50}
        };
        auto it = creditNonQCorr.find(f1.bucket);
        return (it != creditNonQCorr.end()) ? it->second : 0.50;
    }

    return 1.0;
}

// ============================================================================
// Inter-Bucket Gamma Correlation
// ============================================================================
inline double getInterBucketGamma(RiskClass rc, int b1, int b2) {
    if (b1 == b2) return 1.0;

    if (rc == RiskClass::Rates)
        return v26::IR_GAMMA_DIFF_CCY;

    if (rc == RiskClass::CreditQ) {
        if (b1 >= 1 && b1 <= 12 && b2 >= 1 && b2 <= 12)
            return v26::CREDITQ_INTER_BUCKET[b1-1][b2-1];
        return 0.5; // residual
    }

    if (rc == RiskClass::CreditNonQ)
        return v26::CR_GAMMA_DIFF_BUCKET;

    if (rc == RiskClass::Equity) {
        if (b1 >= 1 && b1 <= 12 && b2 >= 1 && b2 <= 12)
            return v26::EQUITY_INTER_BUCKET[b1-1][b2-1];
        return 0.0; // residual
    }

    if (rc == RiskClass::Commodity) {
        if (b1 >= 1 && b1 <= 17 && b2 >= 1 && b2 <= 17)
            return v26::COMMODITY_INTER_BUCKET[b1-1][b2-1];
        return 0.0;
    }

    // FX: no inter-bucket (single bucket)
    return 0.0;
}

// ============================================================================
// Build Factor Metadata
// ============================================================================
using RiskFactorKey = std::tuple<std::string, std::string, int, std::string>;

inline std::vector<FactorMeta> buildFactorMetadata(
    const std::vector<RiskFactorKey>& risk_factors,
    const std::unordered_map<std::string, double>& bucket_sum_sens = {})
{
    std::vector<FactorMeta> metadata;
    metadata.reserve(risk_factors.size());

    for (const auto& [rt, qualifier, bucket, label1] : risk_factors) {
        RiskClass rc = mapRiskTypeToClass(rt);
        bool is_delta = isDeltaRiskType(rt);
        bool is_vega = isVegaRiskType(rt);
        RiskMeasure measure = is_vega ? RiskMeasure::Vega : RiskMeasure::Delta;

        int tenor_idx = -1;
        if (rt == "Risk_IRCurve" || rt == "Risk_IRVol") {
            tenor_idx = v26::tenorIndex(label1);
        }

        // Risk weight
        double weight;
        if (is_vega) {
            weight = getVegaRiskWeight(rt, bucket);
        } else {
            weight = getRiskWeight(rt, qualifier, bucket, label1);
        }

        // Bucket key for grouping
        std::string bucket_key;
        if (rc == RiskClass::Rates || rc == RiskClass::FX) {
            bucket_key = qualifier;
        } else {
            bucket_key = std::to_string(bucket);
        }

        // Concentration factor (from pre-computed sums)
        double cr = 1.0;
        if (rt != "Risk_XCcyBasis") {
            auto it = bucket_sum_sens.find(std::string(riskClassName(rc)) + "|" + bucket_key);
            if (it != bucket_sum_sens.end()) {
                double sum_abs = std::abs(it->second);
                double threshold = 1.0;

                if (is_delta) {
                    if (rc == RiskClass::Rates) threshold = v26::getIRDeltaCT(qualifier);
                    else if (rc == RiskClass::FX) threshold = v26::getFXDeltaCT(qualifier);
                    else if (rc == RiskClass::Equity) {
                        auto ct = v26::EQUITY_DELTA_CT.find(bucket);
                        threshold = (ct != v26::EQUITY_DELTA_CT.end()) ? ct->second : 0.37e9;
                    }
                    else if (rc == RiskClass::Commodity) {
                        auto ct = v26::COMMODITY_DELTA_CT.find(bucket);
                        threshold = (ct != v26::COMMODITY_DELTA_CT.end()) ? ct->second : 52e6;
                    }
                    else if (rc == RiskClass::CreditQ) threshold = v26::getCreditQDeltaCT(bucket);
                    else if (rc == RiskClass::CreditNonQ) threshold = v26::getCreditNonQDeltaCT(bucket);
                } else if (is_vega) {
                    if (rc == RiskClass::Rates) threshold = v26::getIRVegaCT(qualifier);
                    else if (rc == RiskClass::CreditQ) threshold = v26::CREDITQ_VEGA_CT;
                    else if (rc == RiskClass::CreditNonQ) threshold = v26::CREDITNONQ_VEGA_CT;
                    else if (rc == RiskClass::Equity) {
                        auto ct = v26::EQUITY_VEGA_CT.find(bucket);
                        threshold = (ct != v26::EQUITY_VEGA_CT.end()) ? ct->second : 39e6;
                    }
                    else if (rc == RiskClass::Commodity) {
                        auto ct = v26::COMMODITY_VEGA_CT.find(bucket);
                        threshold = (ct != v26::COMMODITY_VEGA_CT.end()) ? ct->second : 69e6;
                    }
                }

                if (threshold > 0) {
                    cr = std::max(1.0, sum_abs / threshold);
                }
            }
        }

        metadata.push_back({
            rc, measure, rt, qualifier, bucket, label1,
            tenor_idx, weight, cr, bucket_key
        });
    }

    return metadata;
}

} // namespace simm
