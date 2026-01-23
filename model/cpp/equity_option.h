#pragma once
#include "market_data.h"
#include <cmath>

namespace simm {

struct EquityOptionTrade {
    double notional = 1e6;
    double strike = 100.0;
    double maturity = 1.0;       // years
    double dividend_yield = 0.02; // continuous dividend yield
    bool is_call = true;
    int equity_bucket = 0;       // SIMM bucket (0-11)
};

// Price an equity option using Black-Scholes with continuous dividend yield
template <typename T>
inline T priceEquityOption(const EquityOptionTrade& trade, const YieldCurve<T>& curve,
                           T spot, T vol) {
    T r = curve.zeroRate(trade.maturity);
    T q = T(trade.dividend_yield);
    double tau = trade.maturity;
    T sqrt_tau = sqrt(T(tau));

    T d1 = (log(spot / T(trade.strike)) + (r - q + T(0.5) * vol * vol) * T(tau)) /
            (vol * sqrt_tau);
    T d2 = d1 - vol * sqrt_tau;

    T df = exp(-r * T(tau));
    T dq = exp(-q * T(tau));

    T price;
    double num_contracts = trade.notional / trade.strike;

    if (trade.is_call) {
        price = T(num_contracts) * (spot * dq * normalCDF(d1) -
                T(trade.strike) * df * normalCDF(d2));
    } else {
        price = T(num_contracts) * (T(trade.strike) * df * normalCDF(-d2) -
                spot * dq * normalCDF(-d1));
    }

    return price;
}

}  // namespace simm
