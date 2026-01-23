#pragma once
#include "market_data.h"
#include <cmath>
#include <string>

namespace simm {

struct XCCYSwapTrade {
    double dom_notional = 1e6;
    double fgn_notional = 800000.0;  // determined by initial FX spot
    double dom_fixed_rate = 0.03;
    double fgn_fixed_rate = 0.02;
    double maturity = 5.0;
    int frequency = 2;               // payments per year (both legs)
    bool exchange_notional = true;   // initial + final notional exchange
    std::string domestic_ccy = "USD";
    std::string foreign_ccy = "EUR";
};

// Price a cross-currency swap
template <typename T>
inline T priceXCCYSwap(const XCCYSwapTrade& trade, const YieldCurve<T>& dom_curve,
                       const YieldCurve<T>& fgn_curve, T fxSpot) {
    double dt = 1.0 / trade.frequency;
    int num_periods = static_cast<int>(trade.maturity * trade.frequency);

    // Domestic leg: receive fixed coupons in domestic currency
    T dom_leg = T(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        T df = dom_curve.discount(t);
        dom_leg = dom_leg + T(trade.dom_notional * trade.dom_fixed_rate * dt) * df;
    }

    // Foreign leg: pay fixed coupons in foreign currency
    T fgn_leg = T(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        T df = fgn_curve.discount(t);
        fgn_leg = fgn_leg + T(trade.fgn_notional * trade.fgn_fixed_rate * dt) * df;
    }

    // Notional exchanges
    if (trade.exchange_notional) {
        T dom_df_mat = dom_curve.discount(trade.maturity);
        T fgn_df_mat = fgn_curve.discount(trade.maturity);

        // At maturity: receive domestic notional, pay foreign notional
        dom_leg = dom_leg + T(trade.dom_notional) * dom_df_mat;
        fgn_leg = fgn_leg + T(trade.fgn_notional) * fgn_df_mat;

        // At inception (t=0): pay domestic notional, receive foreign notional
        dom_leg = dom_leg - T(trade.dom_notional);
        fgn_leg = fgn_leg - T(trade.fgn_notional);
    }

    // NPV in domestic currency
    T npv = dom_leg - fgn_leg * fxSpot;
    return npv;
}

}  // namespace simm
