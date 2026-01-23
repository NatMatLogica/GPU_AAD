#pragma once
#include <array>
#include <cmath>
#include <vector>
#include <string>

namespace simm {

// ISDA SIMM IR Delta: 12 tenor buckets
constexpr int NUM_IR_TENORS = 12;
constexpr std::array<double, NUM_IR_TENORS> IR_TENORS = {
    2.0/52, 1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0
};
// Labels for display
inline const std::array<std::string, NUM_IR_TENORS> IR_TENOR_LABELS = {
    "2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "15y", "30y"
};

// IR Delta risk weights (bps) - SIMM v2.6 approximation for regular volatility currencies
constexpr std::array<double, NUM_IR_TENORS> IR_RISK_WEIGHTS = {
    77.0, 77.0, 74.0, 63.0, 56.0, 52.0, 51.0, 51.0, 51.0, 53.0, 56.0, 64.0
};

// IR intra-bucket correlation (between tenors)
inline double irCorrelation(int i, int j) {
    if (i == j) return 1.0;
    double ti = IR_TENORS[i];
    double tj = IR_TENORS[j];
    double theta = 0.99;  // SIMM parameter
    return std::max(std::exp(-theta * std::abs(std::log(tj / ti))), 0.4);
}

// IR Vega: 6 expiry buckets x same 12 tenor buckets
constexpr int NUM_VEGA_EXPIRIES = 6;
constexpr std::array<double, NUM_VEGA_EXPIRIES> VEGA_EXPIRIES = {
    0.5, 1.0, 3.0, 5.0, 10.0, 30.0
};
constexpr double IR_VEGA_RISK_WEIGHT = 0.21;  // 21% of vega notional

// Equity Delta: 12 buckets by sector
constexpr int NUM_EQ_BUCKETS = 12;
constexpr std::array<double, NUM_EQ_BUCKETS> EQ_DELTA_RISK_WEIGHTS = {
    25.0, 28.0, 29.0, 27.0, 18.0, 21.0, 25.0, 22.0, 27.0, 29.0, 16.0, 25.0
};
constexpr double EQ_VEGA_RISK_WEIGHT = 0.28;  // 28% of vega notional

// Equity intra-bucket correlation
constexpr double EQ_INTRA_CORR = 0.14;

// FX Delta risk weight
constexpr double FX_DELTA_RISK_WEIGHT = 8.2;  // percentage
constexpr double FX_VEGA_RISK_WEIGHT = 0.30;  // 30% of vega notional

// FX correlation between currency pairs
constexpr double FX_CORR = 0.5;

// Inflation risk weight (same structure as IR)
constexpr double INFLATION_RISK_WEIGHT = 63.0;  // bps
constexpr double INFLATION_CORR = 0.33;  // correlation with IR

// SIMM aggregation: within a risk class, the margin is:
// K = sqrt(sum_i sum_j rho_ij * WS_i * WS_j)
// where WS_i = RiskWeight_i * Sensitivity_i
// Then across risk classes: IM = sum of K_class (simplified, ignoring cross-class correlation)

struct SIMMResults {
    double ir_delta_margin = 0.0;
    double ir_vega_margin = 0.0;
    double eq_delta_margin = 0.0;
    double eq_vega_margin = 0.0;
    double fx_delta_margin = 0.0;
    double fx_vega_margin = 0.0;
    double inflation_margin = 0.0;
    double total_margin = 0.0;

    void computeTotal() {
        total_margin = ir_delta_margin + ir_vega_margin +
                       eq_delta_margin + eq_vega_margin +
                       fx_delta_margin + fx_vega_margin +
                       inflation_margin;
    }
};

// Aggregate IR delta sensitivities into SIMM margin
inline double aggregateIRDelta(const std::array<double, NUM_IR_TENORS>& sensitivities) {
    double sum = 0.0;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        double ws_i = IR_RISK_WEIGHTS[i] * sensitivities[i];
        for (int j = 0; j < NUM_IR_TENORS; ++j) {
            double ws_j = IR_RISK_WEIGHTS[j] * sensitivities[j];
            sum += irCorrelation(i, j) * ws_i * ws_j;
        }
    }
    return std::sqrt(std::max(sum, 0.0));
}

// Aggregate equity delta sensitivities (single bucket)
inline double aggregateEQDelta(double sensitivity, int bucket = 0) {
    double ws = EQ_DELTA_RISK_WEIGHTS[bucket] * sensitivity;
    return std::abs(ws);
}

// Aggregate FX delta sensitivities
inline double aggregateFXDelta(const std::vector<double>& sensitivities) {
    double sum = 0.0;
    int n = static_cast<int>(sensitivities.size());
    for (int i = 0; i < n; ++i) {
        double ws_i = FX_DELTA_RISK_WEIGHT * sensitivities[i];
        for (int j = 0; j < n; ++j) {
            double ws_j = FX_DELTA_RISK_WEIGHT * sensitivities[j];
            double corr = (i == j) ? 1.0 : FX_CORR;
            sum += corr * ws_i * ws_j;
        }
    }
    return std::sqrt(std::max(sum, 0.0));
}

// Aggregate inflation sensitivities
inline double aggregateInflation(const std::array<double, NUM_IR_TENORS>& sensitivities) {
    double sum = 0.0;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        double ws_i = INFLATION_RISK_WEIGHT * sensitivities[i];
        for (int j = 0; j < NUM_IR_TENORS; ++j) {
            double ws_j = INFLATION_RISK_WEIGHT * sensitivities[j];
            double corr = (i == j) ? 1.0 : INFLATION_CORR;
            sum += corr * ws_i * ws_j;
        }
    }
    return std::sqrt(std::max(sum, 0.0));
}

}  // namespace simm
