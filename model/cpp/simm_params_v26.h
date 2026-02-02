// ISDA SIMM v2.6 Parameters
// Source: https://www.isda.org/a/b4ugE/ISDA-SIMM_v2.6_PUBLIC.pdf
// Transcribed from Weights_and_Corr/v2_6.py
// Version: 1.0.0
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace simm::v26 {

// ============================================================================
// Tenor Structure
// ============================================================================
constexpr int NUM_TENORS = 12;
constexpr int NUM_VEGA_EXPIRIES = 6;

inline const std::array<std::string, 12> TENOR_LABELS = {
    "2w","1m","3m","6m","1y","2y","3y","5y","10y","15y","20y","30y"
};

inline const std::array<double, 12> TENOR_YEARS = {
    2.0/52, 1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0
};

inline int tenorIndex(const std::string& label) {
    for (int i = 0; i < NUM_TENORS; ++i) {
        if (TENOR_LABELS[i] == label) return i;
    }
    return -1;
}

// ============================================================================
// Currency Buckets
// ============================================================================
inline const std::unordered_set<std::string> REG_VOL_CCY = {
    "USD","EUR","GBP","CHF","AUD","NZD","CAD","SEK","NOK","DKK","HKD","KRW","SGD","TWD"
};

inline const std::unordered_set<std::string> LOW_VOL_CCY = {"JPY"};

inline const std::unordered_set<std::string> HIGH_VOL_FX_CCY = {"BRL","RUB","TRY"};

inline bool isRegVolCcy(const std::string& ccy) { return REG_VOL_CCY.count(ccy) > 0; }
inline bool isLowVolCcy(const std::string& ccy) { return LOW_VOL_CCY.count(ccy) > 0; }
inline bool isHighVolCcy(const std::string& ccy) { return HIGH_VOL_FX_CCY.count(ccy) > 0; }

// ============================================================================
// IR Risk Weights (by tenor, by volatility class)
// ============================================================================
constexpr std::array<double, 12> REG_VOL_RW = {109,105,90,71,66,66,64,60,60,61,61,67};
constexpr std::array<double, 12> LOW_VOL_RW = {15,18,9,11,13,15,19,23,23,22,22,23};
constexpr std::array<double, 12> HIGH_VOL_RW = {163,109,87,89,102,96,101,97,97,102,106,101};

inline double getIRRiskWeight(const std::string& ccy, int tenorIdx) {
    if (tenorIdx < 0 || tenorIdx >= NUM_TENORS) return 0.0;
    if (isLowVolCcy(ccy)) return LOW_VOL_RW[tenorIdx];
    if (isRegVolCcy(ccy)) return REG_VOL_RW[tenorIdx];
    return HIGH_VOL_RW[tenorIdx]; // high vol
}

inline double getIRRiskWeight(const std::string& ccy, const std::string& tenorLabel) {
    return getIRRiskWeight(ccy, tenorIndex(tenorLabel));
}

// Inflation and XCcy Basis
constexpr double INFLATION_RW = 61.0;
constexpr double CCY_BASIS_SWAP_SPREAD_RW = 21.0;

// Historical Volatility Ratio
constexpr double IR_HVR = 0.47;

// Vega Risk Weight
constexpr double IR_VRW = 0.23;

// ============================================================================
// IR Tenor Correlation Matrix (12x12)
// ============================================================================
constexpr std::array<std::array<double, 12>, 12> IR_CORR = {{
    {{1.00, 0.77, 0.67, 0.59, 0.48, 0.39, 0.34, 0.30, 0.25, 0.23, 0.21, 0.20}},
    {{0.77, 1.00, 0.84, 0.74, 0.56, 0.43, 0.36, 0.31, 0.26, 0.21, 0.19, 0.19}},
    {{0.67, 0.84, 1.00, 0.88, 0.69, 0.55, 0.47, 0.40, 0.34, 0.27, 0.25, 0.25}},
    {{0.59, 0.74, 0.88, 1.00, 0.86, 0.73, 0.65, 0.57, 0.49, 0.40, 0.38, 0.37}},
    {{0.48, 0.56, 0.69, 0.86, 1.00, 0.94, 0.87, 0.79, 0.68, 0.60, 0.57, 0.55}},
    {{0.39, 0.43, 0.55, 0.73, 0.94, 1.00, 0.96, 0.91, 0.80, 0.74, 0.70, 0.69}},
    {{0.34, 0.36, 0.47, 0.65, 0.87, 0.96, 1.00, 0.97, 0.88, 0.81, 0.77, 0.76}},
    {{0.30, 0.31, 0.40, 0.57, 0.79, 0.91, 0.97, 1.00, 0.95, 0.90, 0.86, 0.85}},
    {{0.25, 0.26, 0.34, 0.49, 0.68, 0.80, 0.88, 0.95, 1.00, 0.97, 0.94, 0.94}},
    {{0.23, 0.21, 0.27, 0.40, 0.60, 0.74, 0.81, 0.90, 0.97, 1.00, 0.98, 0.97}},
    {{0.21, 0.19, 0.25, 0.38, 0.57, 0.70, 0.77, 0.86, 0.94, 0.98, 1.00, 0.99}},
    {{0.20, 0.19, 0.25, 0.37, 0.55, 0.69, 0.76, 0.85, 0.94, 0.97, 0.99, 1.00}},
}};

// Sub-curve correlations
constexpr double SUB_CURVES_CORR = 0.993;
constexpr double INFLATION_CORR = 0.24;
constexpr double CCY_BASIS_SPREAD_CORR = 0.04;

// Inter-bucket (cross-currency) gamma
constexpr double IR_GAMMA_DIFF_CCY = 0.32;

// ============================================================================
// Credit Qualifying
// ============================================================================
inline const std::unordered_map<int, double> CREDITQ_RW = {
    {1,75},{2,90},{3,84},{4,54},{5,62},{6,48},{7,185},{8,343},{9,255},{10,250},{11,214},{12,173},{0,343}
};
constexpr double CREDITQ_VRW = 0.76;
constexpr double BASE_CORR_WEIGHT = 10.0;

// Intra-bucket correlations: [same_issuer_same_tenor, same_issuer_diff_tenor, diff_issuer_same_tenor, diff_issuer_diff_tenor]
// creditQ_corr = [0.93, 0.46, 0.5, 0.29]
inline const std::array<double, 4> CREDITQ_CORR = {0.93, 0.46, 0.5, 0.29};

// Inter-bucket correlations (12x12)
constexpr std::array<std::array<double, 12>, 12> CREDITQ_INTER_BUCKET = {{
    {{1.00, 0.38, 0.38, 0.35, 0.37, 0.34, 0.42, 0.32, 0.34, 0.33, 0.34, 0.33}},
    {{0.38, 1.00, 0.48, 0.46, 0.48, 0.46, 0.39, 0.40, 0.41, 0.41, 0.43, 0.40}},
    {{0.38, 0.48, 1.00, 0.50, 0.51, 0.50, 0.40, 0.39, 0.45, 0.44, 0.47, 0.42}},
    {{0.35, 0.46, 0.50, 1.00, 0.50, 0.50, 0.37, 0.37, 0.41, 0.43, 0.45, 0.40}},
    {{0.37, 0.48, 0.51, 0.50, 1.00, 0.50, 0.39, 0.38, 0.43, 0.43, 0.46, 0.42}},
    {{0.34, 0.46, 0.50, 0.50, 0.50, 1.00, 0.37, 0.35, 0.39, 0.41, 0.44, 0.41}},
    {{0.42, 0.39, 0.40, 0.37, 0.39, 0.37, 1.00, 0.33, 0.37, 0.37, 0.35, 0.35}},
    {{0.32, 0.40, 0.39, 0.37, 0.38, 0.35, 0.33, 1.00, 0.36, 0.37, 0.37, 0.36}},
    {{0.34, 0.41, 0.45, 0.41, 0.43, 0.39, 0.37, 0.36, 1.00, 0.41, 0.40, 0.38}},
    {{0.33, 0.41, 0.44, 0.43, 0.43, 0.41, 0.37, 0.37, 0.41, 1.00, 0.41, 0.39}},
    {{0.34, 0.43, 0.47, 0.45, 0.46, 0.44, 0.35, 0.37, 0.40, 0.41, 1.00, 0.40}},
    {{0.33, 0.40, 0.42, 0.40, 0.42, 0.41, 0.35, 0.36, 0.38, 0.39, 0.40, 1.00}},
}};

// ============================================================================
// Credit Non-Qualifying
// ============================================================================
inline const std::unordered_map<int, double> CREDITNONQ_RW = {
    {1,280},{2,1300},{0,1300}
};
constexpr double CREDITNONQ_VRW = 0.76;
// Intra-bucket: [same_issuer_same_tenor, same_issuer_diff_tenor, diff_issuer]
inline const std::array<double, 3> CREDITNONQ_CORR = {0.83, 0.32, 0.5};
constexpr double CR_GAMMA_DIFF_BUCKET = 0.43;

// ============================================================================
// Equity
// ============================================================================
inline const std::unordered_map<int, double> EQUITY_RW = {
    {1,30},{2,33},{3,36},{4,29},{5,26},{6,25},{7,34},{8,28},{9,36},{10,50},{11,19},{12,19},{0,50}
};
constexpr double EQUITY_HVR = 0.60;
constexpr double EQUITY_VRW = 0.45;
constexpr double EQUITY_VRW_BUCKET_12 = 0.96;

inline const std::unordered_map<int, double> EQUITY_CORR = {
    {1,0.18},{2,0.20},{3,0.28},{4,0.24},{5,0.25},{6,0.36},{7,0.35},{8,0.37},
    {9,0.23},{10,0.27},{11,0.45},{12,0.45},{0,0.0}
};

constexpr std::array<std::array<double, 12>, 12> EQUITY_INTER_BUCKET = {{
    {{1.00, 0.18, 0.19, 0.19, 0.14, 0.16, 0.15, 0.16, 0.18, 0.12, 0.19, 0.19}},
    {{0.18, 1.00, 0.22, 0.21, 0.15, 0.18, 0.17, 0.19, 0.20, 0.14, 0.21, 0.21}},
    {{0.19, 0.22, 1.00, 0.22, 0.13, 0.16, 0.18, 0.17, 0.22, 0.13, 0.20, 0.20}},
    {{0.19, 0.21, 0.22, 1.00, 0.17, 0.22, 0.22, 0.23, 0.22, 0.17, 0.26, 0.26}},
    {{0.14, 0.15, 0.13, 0.17, 1.00, 0.29, 0.26, 0.29, 0.14, 0.24, 0.32, 0.32}},
    {{0.16, 0.18, 0.16, 0.22, 0.29, 1.00, 0.34, 0.36, 0.17, 0.30, 0.39, 0.39}},
    {{0.15, 0.17, 0.18, 0.22, 0.26, 0.34, 1.00, 0.33, 0.16, 0.28, 0.36, 0.36}},
    {{0.16, 0.19, 0.17, 0.23, 0.29, 0.36, 0.33, 1.00, 0.17, 0.29, 0.40, 0.40}},
    {{0.18, 0.20, 0.22, 0.22, 0.14, 0.17, 0.16, 0.17, 1.00, 0.13, 0.21, 0.21}},
    {{0.12, 0.14, 0.13, 0.17, 0.24, 0.30, 0.28, 0.29, 0.13, 1.00, 0.30, 0.30}},
    {{0.19, 0.21, 0.20, 0.26, 0.32, 0.39, 0.36, 0.40, 0.21, 0.30, 1.00, 0.45}},
    {{0.19, 0.21, 0.20, 0.26, 0.32, 0.39, 0.36, 0.40, 0.21, 0.30, 0.45, 1.00}},
}};

// ============================================================================
// Commodity
// ============================================================================
inline const std::unordered_map<int, double> COMMODITY_RW = {
    {1,48},{2,29},{3,33},{4,25},{5,35},{6,30},{7,60},{8,52},{9,68},{10,63},
    {11,21},{12,21},{13,15},{14,16},{15,13},{16,68},{17,17}
};
constexpr double COMMODITY_HVR = 0.74;
constexpr double COMMODITY_VRW = 0.55;

inline const std::unordered_map<int, double> COMMODITY_CORR = {
    {1,0.83},{2,0.97},{3,0.93},{4,0.97},{5,0.98},{6,0.90},{7,0.98},{8,0.49},
    {9,0.80},{10,0.46},{11,0.58},{12,0.53},{13,0.62},{14,0.16},{15,0.18},{16,0.00},{17,0.38}
};

constexpr std::array<std::array<double, 17>, 17> COMMODITY_INTER_BUCKET = {{
    {{1.00, 0.22, 0.18, 0.21, 0.20, 0.24,  0.49,  0.16,  0.38, 0.14, 0.10,  0.02, 0.12, 0.11,  0.02, 0.00, 0.17}},
    {{0.22, 1.00, 0.92, 0.90, 0.88, 0.25,  0.08,  0.19,  0.17, 0.17, 0.42,  0.28, 0.36, 0.27,  0.20, 0.00, 0.64}},
    {{0.18, 0.92, 1.00, 0.87, 0.84, 0.16,  0.07,  0.15,  0.10, 0.18, 0.33,  0.22, 0.27, 0.23,  0.16, 0.00, 0.54}},
    {{0.21, 0.90, 0.87, 1.00, 0.77, 0.19,  0.11,  0.18,  0.16, 0.14, 0.32,  0.22, 0.28, 0.22,  0.11, 0.00, 0.58}},
    {{0.20, 0.88, 0.84, 0.77, 1.00, 0.19,  0.09,  0.12,  0.13, 0.18, 0.42,  0.34, 0.32, 0.29,  0.13, 0.00, 0.59}},
    {{0.24, 0.25, 0.16, 0.19, 0.19, 1.00,  0.31,  0.62,  0.23, 0.10, 0.21,  0.05, 0.18, 0.10,  0.08, 0.00, 0.28}},
    {{0.49, 0.08, 0.07, 0.11, 0.09, 0.31,  1.00,  0.21,  0.79, 0.17, 0.10, -0.08, 0.10, 0.07, -0.02, 0.00, 0.13}},
    {{0.16, 0.19, 0.15, 0.18, 0.12, 0.62,  0.21,  1.00,  0.16, 0.08, 0.13, -0.07, 0.07, 0.05,  0.02, 0.00, 0.19}},
    {{0.38, 0.17, 0.10, 0.16, 0.13, 0.23,  0.79,  0.16,  1.00, 0.15, 0.09, -0.06, 0.06, 0.06,  0.01, 0.00, 0.16}},
    {{0.14, 0.17, 0.18, 0.14, 0.18, 0.10,  0.17,  0.08,  0.15, 1.00, 0.16,  0.09, 0.14, 0.09,  0.03, 0.00, 0.11}},
    {{0.10, 0.42, 0.33, 0.32, 0.42, 0.21,  0.10,  0.13,  0.09, 0.16, 1.00,  0.36, 0.30, 0.25,  0.18, 0.00, 0.37}},
    {{0.02, 0.28, 0.22, 0.22, 0.34, 0.05, -0.08, -0.07, -0.06, 0.09, 0.36,  1.00, 0.20, 0.18,  0.11, 0.00, 0.26}},
    {{0.12, 0.36, 0.27, 0.28, 0.32, 0.18,  0.10,  0.07,  0.06, 0.14, 0.30,  0.20, 1.00, 0.28,  0.19, 0.00, 0.39}},
    {{0.11, 0.27, 0.23, 0.22, 0.29, 0.10,  0.07,  0.05,  0.06, 0.09, 0.25,  0.18, 0.28, 1.00,  0.13, 0.00, 0.26}},
    {{0.02, 0.20, 0.16, 0.11, 0.13, 0.08, -0.02,  0.02,  0.01, 0.03, 0.18,  0.11, 0.19, 0.13,  1.00, 0.00, 0.21}},
    {{0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  0.00,  0.00,  0.00, 0.00, 0.00,  0.00, 0.00, 0.00,  0.00, 1.00, 0.00}},
    {{0.17, 0.64, 0.54, 0.58, 0.59, 0.28,  0.13,  0.19,  0.16, 0.11, 0.37,  0.26, 0.39, 0.26,  0.21, 0.00, 1.00}},
}};

// ============================================================================
// FX
// ============================================================================
// Risk weights depend on volatility category of both currencies
// fx_rw[calc_ccy_group][qual_group]
inline double getFXRiskWeight(const std::string& calcCcy, const std::string& qualifier) {
    bool calcHigh = isHighVolCcy(calcCcy);
    bool qualHigh = isHighVolCcy(qualifier);
    if (!calcHigh && !qualHigh) return 7.4;
    if (calcHigh && qualHigh)   return 21.4;
    return 14.7; // one high, one regular
}

constexpr double FX_HVR = 0.57;
constexpr double FX_VRW = 0.48;

// FX delta correlations depend on calculation currency volatility group
inline double getFXDeltaCorrelation(const std::string& calcCcy,
                                     const std::string& q1, const std::string& q2) {
    if (q1 == q2) return 1.0;
    bool calcHigh = isHighVolCcy(calcCcy);
    bool q1High = isHighVolCcy(q1);
    bool q2High = isHighVolCcy(q2);
    if (calcHigh) {
        // High vol calc ccy correlations
        if (!q1High && !q2High) return 0.88;
        if (q1High && q2High)   return 0.50;
        return 0.72;
    } else {
        // Regular vol calc ccy correlations
        if (!q1High && !q2High) return 0.50;
        if (q1High && q2High)   return -0.05;
        return 0.25;
    }
}

constexpr double FX_VEGA_CORR = 0.5;

// ============================================================================
// Concentration Thresholds - Delta
// ============================================================================
inline const std::unordered_map<std::string, double> IR_DELTA_CT = {
    {"USD",330},{"EUR",330},{"GBP",330},
    {"AUD",130},{"CAD",130},{"CHF",130},{"DKK",130},{"HKD",130},
    {"KRW",130},{"NOK",130},{"NZD",130},{"SEK",130},{"SGD",130},{"TWD",130},
    {"JPY",61},{"Others",30}
};

inline double getIRDeltaCT(const std::string& ccy) {
    auto it = IR_DELTA_CT.find(ccy);
    return (it != IR_DELTA_CT.end()) ? it->second * 1e6 : 30e6;
}

// Credit Delta CT (in billions)
inline double getCreditQDeltaCT(int bucket) {
    // Sovereigns: buckets 1,7 -> 1.0B; Corporate: others -> 0.17B; Residual 0: 0.17B
    if (bucket == 1 || bucket == 7) return 1.00e9;
    return 0.17e9;
}

inline double getCreditNonQDeltaCT(int bucket) {
    if (bucket == 1) return 9.5e9;
    return 0.5e9; // bucket 2 and residual
}

inline const std::unordered_map<int, double> EQUITY_DELTA_CT = {
    {1,3e9},{2,3e9},{3,3e9},{4,3e9},
    {5,12e9},{6,12e9},{7,12e9},{8,12e9},
    {9,0.64e9},{10,0.37e9},
    {11,810e9},{12,810e9},{0,0.37e9}
};

inline const std::unordered_map<int, double> COMMODITY_DELTA_CT = {
    {1,310e6},{2,2100e6},{3,1700e6},{4,1700e6},{5,1700e6},
    {6,2800e6},{7,2800e6},{8,2700e6},{9,2700e6},{10,52e6},
    {11,530e6},{12,1300e6},{13,100e6},{14,100e6},{15,100e6},{16,52e6},{17,4000e6}
};

// FX Delta CT by category
inline const std::unordered_set<std::string> FX_CATEGORY1 = {
    "USD","EUR","JPY","GBP","AUD","CHF","CAD"
};
inline const std::unordered_set<std::string> FX_CATEGORY2 = {
    "BRL","CNY","HKD","INR","KRW","MXN","NOK","NZD","RUB","SEK","SGD","TRY","ZAR"
};

inline double getFXDeltaCT(const std::string& ccy) {
    if (FX_CATEGORY1.count(ccy)) return 3300e6;
    if (FX_CATEGORY2.count(ccy)) return 880e6;
    return 170e6;
}

// ============================================================================
// Concentration Thresholds - Vega
// ============================================================================
inline const std::unordered_map<std::string, double> IR_VEGA_CT = {
    {"USD",4900e6},{"EUR",4900e6},{"GBP",4900e6},
    {"AUD",520e6},{"CAD",520e6},{"CHF",520e6},{"DKK",520e6},{"HKD",520e6},
    {"KRW",520e6},{"NOK",520e6},{"NZD",520e6},{"SEK",520e6},{"SGD",520e6},{"TWD",520e6},
    {"JPY",970e6}
};

inline double getIRVegaCT(const std::string& ccy) {
    auto it = IR_VEGA_CT.find(ccy);
    return (it != IR_VEGA_CT.end()) ? it->second : 74e6;
}

constexpr double CREDITQ_VEGA_CT = 360e6;
constexpr double CREDITNONQ_VEGA_CT = 70e6;

inline const std::unordered_map<int, double> EQUITY_VEGA_CT = {
    {1,210e6},{2,210e6},{3,210e6},{4,210e6},
    {5,1300e6},{6,1300e6},{7,1300e6},{8,1300e6},
    {9,39e6},{10,190e6},
    {11,6400e6},{12,6400e6},{0,39e6}
};

inline const std::unordered_map<int, double> COMMODITY_VEGA_CT = {
    {1,390e6},{2,2900e6},{3,310e6},{4,310e6},{5,310e6},
    {6,6300e6},{7,6300e6},{8,1200e6},{9,1200e6},{10,120e6},
    {11,390e6},{12,1300e6},{13,590e6},{14,590e6},{15,590e6},{16,69e6},{17,69e6}
};

// ============================================================================
// Cross-Risk-Class Correlation (PSI matrix, 6x6)
// Order: Rates, CreditQ, CreditNonQ, Equity, Commodity, FX
// ============================================================================
constexpr std::array<std::array<double, 6>, 6> PSI = {{
    {{1.00, 0.04, 0.04, 0.07, 0.37, 0.14}},
    {{0.04, 1.00, 0.54, 0.70, 0.27, 0.37}},
    {{0.04, 0.54, 1.00, 0.46, 0.24, 0.15}},
    {{0.07, 0.70, 0.46, 1.00, 0.35, 0.39}},
    {{0.37, 0.27, 0.24, 0.35, 1.00, 0.35}},
    {{0.14, 0.37, 0.15, 0.39, 0.35, 1.00}},
}};

} // namespace simm::v26
