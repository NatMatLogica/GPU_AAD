#pragma once
#include <array>
#include <cmath>
#include <algorithm>
#include "simm_config.h"

namespace simm {

// Yield curve: zero rates at SIMM tenors, linear interpolation, continuous compounding
template <typename T = double>
struct YieldCurve {
    std::array<T, NUM_IR_TENORS> zero_rates{};  // continuously compounded zero rates

    // Interpolate zero rate at arbitrary time t
    T zeroRate(double t) const {
        if (t <= IR_TENORS[0]) return zero_rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return zero_rates[NUM_IR_TENORS - 1];
        for (int i = 0; i < NUM_IR_TENORS - 1; ++i) {
            if (t >= IR_TENORS[i] && t <= IR_TENORS[i + 1]) {
                double w = (t - IR_TENORS[i]) / (IR_TENORS[i + 1] - IR_TENORS[i]);
                return zero_rates[i] * (1.0 - w) + zero_rates[i + 1] * w;
            }
        }
        return zero_rates[NUM_IR_TENORS - 1];
    }

    // Discount factor at time t
    T discount(double t) const {
        T r = zeroRate(t);
        return exp(-r * t);
    }

    // Forward rate between t1 and t2
    T forwardRate(double t1, double t2) const {
        if (t2 <= t1) return zeroRate(t1);
        T df1 = discount(t1);
        T df2 = discount(t2);
        return log(df1 / df2) / (t2 - t1);
    }

    // Bump a single tenor bucket by amount (for B&R sensitivities)
    YieldCurve<T> bumped(int tenor_idx, T amount) const {
        YieldCurve<T> result = *this;
        result.zero_rates[tenor_idx] = result.zero_rates[tenor_idx] + amount;
        return result;
    }
};

// Vol surface: flat or term-structure vols by expiry
template <typename T = double>
struct VolSurface {
    std::array<T, NUM_VEGA_EXPIRIES> vols{};  // vols at expiry buckets

    // Interpolate vol at arbitrary expiry
    T vol(double expiry) const {
        if (expiry <= VEGA_EXPIRIES[0]) return vols[0];
        if (expiry >= VEGA_EXPIRIES[NUM_VEGA_EXPIRIES - 1]) return vols[NUM_VEGA_EXPIRIES - 1];
        for (int i = 0; i < NUM_VEGA_EXPIRIES - 1; ++i) {
            if (expiry >= VEGA_EXPIRIES[i] && expiry <= VEGA_EXPIRIES[i + 1]) {
                double w = (expiry - VEGA_EXPIRIES[i]) / (VEGA_EXPIRIES[i + 1] - VEGA_EXPIRIES[i]);
                return vols[i] * (1.0 - w) + vols[i + 1] * w;
            }
        }
        return vols[NUM_VEGA_EXPIRIES - 1];
    }

    // Bump vol at a specific expiry bucket
    VolSurface<T> bumped(int expiry_idx, T amount) const {
        VolSurface<T> result = *this;
        result.vols[expiry_idx] = result.vols[expiry_idx] + amount;
        return result;
    }
};

// Inflation curve: CPI index levels and zero-coupon inflation rates at tenors
template <typename T = double>
struct InflationCurve {
    T base_cpi = T(100.0);  // CPI at time 0
    std::array<T, NUM_IR_TENORS> inflation_rates{};  // zero-coupon inflation rates

    // Projected CPI at time t
    T projectedCPI(double t) const {
        T rate = inflationRate(t);
        return base_cpi * exp(rate * t);
    }

    // Interpolate inflation rate at arbitrary time t
    T inflationRate(double t) const {
        if (t <= IR_TENORS[0]) return inflation_rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return inflation_rates[NUM_IR_TENORS - 1];
        for (int i = 0; i < NUM_IR_TENORS - 1; ++i) {
            if (t >= IR_TENORS[i] && t <= IR_TENORS[i + 1]) {
                double w = (t - IR_TENORS[i]) / (IR_TENORS[i + 1] - IR_TENORS[i]);
                return inflation_rates[i] * (1.0 - w) + inflation_rates[i + 1] * w;
            }
        }
        return inflation_rates[NUM_IR_TENORS - 1];
    }

    // Bump inflation rate at a specific tenor
    InflationCurve<T> bumped(int tenor_idx, T amount) const {
        InflationCurve<T> result = *this;
        result.inflation_rates[tenor_idx] = result.inflation_rates[tenor_idx] + amount;
        return result;
    }
};

// FX Market: spot rate and curves for both currencies
template <typename T = double>
struct FXMarket {
    T spot = T(1.0);                // FX spot rate (domestic per foreign)
    YieldCurve<T> domestic_curve;
    YieldCurve<T> foreign_curve;

    // Bump spot rate
    FXMarket<T> bumpedSpot(T amount) const {
        FXMarket<T> result = *this;
        result.spot = result.spot + amount;
        return result;
    }
};

// Helper: cumulative normal distribution
template <typename T>
T normalCDF(T x) {
    return T(0.5) * (T(1.0) + erf(x / sqrt(T(2.0))));
}

// Helper: standard normal PDF
template <typename T>
T normalPDF(T x) {
    return exp(-T(0.5) * x * x) / sqrt(T(2.0) * T(M_PI));
}

}  // namespace simm
