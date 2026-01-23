#pragma once
#include "market_data.h"
#include <cmath>

namespace simm {

struct IRSwapTrade {
    double notional = 1e6;
    double fixed_rate = 0.03;     // continuously compounded fixed rate
    double maturity = 5.0;        // years
    int frequency = 2;            // payments per year (both legs)
    bool payer = true;            // true = pay fixed, receive floating
};

// Price a vanilla IRS: fixed vs floating, ACT/365, continuous compounding
template <typename T>
inline T priceVanillaIRS(const IRSwapTrade& trade, const YieldCurve<T>& curve) {
    double dt = 1.0 / trade.frequency;
    int num_periods = static_cast<int>(trade.maturity * trade.frequency);

    T fixed_leg = T(0.0);
    T floating_leg = T(0.0);

    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        T df = curve.discount(t);

        // Fixed leg: notional * fixed_rate * dt * DF(t)
        fixed_leg = fixed_leg + T(trade.notional * trade.fixed_rate * dt) * df;

        // Floating leg: notional * forward_rate * dt * DF(t)
        double t_prev = (i - 1) * dt;
        T fwd = curve.forwardRate(t_prev, t);
        floating_leg = floating_leg + T(trade.notional * dt) * fwd * df;
    }

    T npv = floating_leg - fixed_leg;
    if (!trade.payer) {
        npv = fixed_leg - floating_leg;
    }
    return npv;
}

}  // namespace simm
