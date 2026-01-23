#pragma once
#include "market_data.h"
#include <cmath>

namespace simm {

struct InflationSwapTrade {
    double notional = 1e6;
    double fixed_rate = 0.025;   // fixed inflation rate (zero-coupon)
    double maturity = 5.0;       // years
};

// Price a zero-coupon inflation swap
// Fixed leg pays: N * (exp(K*T) - 1) * DF(T)
// Inflation leg pays: N * (CPI_T/CPI_0 - 1) * DF(T)
// NPV = Inflation leg - Fixed leg (from inflation receiver perspective)
template <typename T>
inline T priceInflationSwap(const InflationSwapTrade& trade, const YieldCurve<T>& curve,
                            const InflationCurve<T>& inflation) {
    double tau = trade.maturity;
    T df = curve.discount(tau);

    // Fixed leg: N * (exp(K*T) - 1) * DF(T)
    T fixed_leg = T(trade.notional) * (exp(T(trade.fixed_rate) * T(tau)) - T(1.0)) * df;

    // Inflation leg: N * (CPI_T/CPI_0 - 1) * DF(T)
    T cpi_ratio = inflation.projectedCPI(tau) / inflation.base_cpi;
    T inflation_leg = T(trade.notional) * (cpi_ratio - T(1.0)) * df;

    // NPV from inflation receiver's perspective
    return inflation_leg - fixed_leg;
}

}  // namespace simm
