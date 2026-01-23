#pragma once
#include "market_data.h"
#include <cmath>
#include <string>

namespace simm {

struct FXOptionTrade {
    double notional = 1e6;        // in domestic currency
    double strike = 1.3;          // domestic per foreign
    double maturity = 1.0;        // years
    bool is_call = true;          // call on foreign currency
    std::string domestic_ccy = "USD";
    std::string foreign_ccy = "EUR";
};

// Price an FX option using Garman-Kohlhagen
template <typename T>
inline T priceFXOption(const FXOptionTrade& trade, T spot, T vol,
                       const YieldCurve<T>& dom_curve, const YieldCurve<T>& fgn_curve) {
    T rd = dom_curve.zeroRate(trade.maturity);
    T rf = fgn_curve.zeroRate(trade.maturity);
    double tau = trade.maturity;
    T sqrt_tau = sqrt(T(tau));

    T d1 = (log(spot / T(trade.strike)) + (rd - rf + T(0.5) * vol * vol) * T(tau)) /
            (vol * sqrt_tau);
    T d2 = d1 - vol * sqrt_tau;

    T df_dom = exp(-rd * T(tau));
    T df_fgn = exp(-rf * T(tau));

    // Notional in foreign currency units
    double fgn_notional = trade.notional / trade.strike;

    T price;
    if (trade.is_call) {
        price = T(fgn_notional) * (spot * df_fgn * normalCDF(d1) -
                T(trade.strike) * df_dom * normalCDF(d2));
    } else {
        price = T(fgn_notional) * (T(trade.strike) * df_dom * normalCDF(-d2) -
                spot * df_fgn * normalCDF(-d1));
    }

    return price;
}

}  // namespace simm
