#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
#include <variant>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "simm_config.h"
#include "market_data.h"
#include "vanilla_irs.h"
#include "equity_option.h"
#include "inflation_swap.h"
#include "fx_option.h"
#include "xccy_swap.h"

using namespace simm;
using Trade = std::variant<IRSwapTrade, EquityOptionTrade, InflationSwapTrade,
                           FXOptionTrade, XCCYSwapTrade>;

// Market data (global for simplicity in bump-and-revalue)
struct MarketEnvironment {
    YieldCurve<double> usd_curve;
    YieldCurve<double> eur_curve;
    InflationCurve<double> inflation;
    double equity_spot = 100.0;
    double equity_vol = 0.25;
    double fx_spot = 1.10;      // EURUSD
    double fx_vol = 0.12;
};

MarketEnvironment buildMarket() {
    MarketEnvironment mkt;
    // USD curve: upward sloping from 4% to 5%
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        mkt.usd_curve.zero_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
    }
    // EUR curve: flat at 3%
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        mkt.eur_curve.zero_rates[i] = 0.03 + 0.005 * IR_TENORS[i] / 30.0;
    }
    // Inflation curve: ~2.5% across tenors
    mkt.inflation.base_cpi = 100.0;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        mkt.inflation.inflation_rates[i] = 0.025 + 0.002 * IR_TENORS[i] / 30.0;
    }
    return mkt;
}

double priceTrade(const Trade& trade, const MarketEnvironment& mkt) {
    return std::visit([&](const auto& t) -> double {
        using TradeType = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<TradeType, IRSwapTrade>) {
            return priceVanillaIRS(t, mkt.usd_curve);
        } else if constexpr (std::is_same_v<TradeType, EquityOptionTrade>) {
            return priceEquityOption(t, mkt.usd_curve, mkt.equity_spot, mkt.equity_vol);
        } else if constexpr (std::is_same_v<TradeType, InflationSwapTrade>) {
            return priceInflationSwap(t, mkt.usd_curve, mkt.inflation);
        } else if constexpr (std::is_same_v<TradeType, FXOptionTrade>) {
            return priceFXOption(t, mkt.fx_spot, mkt.fx_vol, mkt.usd_curve, mkt.eur_curve);
        } else if constexpr (std::is_same_v<TradeType, XCCYSwapTrade>) {
            return priceXCCYSwap(t, mkt.usd_curve, mkt.eur_curve, mkt.fx_spot);
        }
        return 0.0;
    }, trade);
}

std::vector<Trade> generatePortfolio(int num_trades, std::mt19937& rng) {
    std::vector<Trade> portfolio;
    portfolio.reserve(num_trades);

    std::uniform_int_distribution<int> type_dist(0, 4);
    std::uniform_real_distribution<double> maturity_dist(1.0, 10.0);
    std::uniform_real_distribution<double> notional_dist(1e6, 50e6);
    std::uniform_int_distribution<int> bool_dist(0, 1);

    for (int i = 0; i < num_trades; ++i) {
        int type = type_dist(rng);
        double notional = notional_dist(rng);
        double maturity = maturity_dist(rng);

        switch (type) {
            case 0: {
                IRSwapTrade t;
                t.notional = notional;
                t.fixed_rate = 0.03 + 0.02 * (rng() % 100) / 100.0;
                t.maturity = maturity;
                t.frequency = 2;
                t.payer = bool_dist(rng);
                portfolio.emplace_back(t);
                break;
            }
            case 1: {
                EquityOptionTrade t;
                t.notional = notional;
                t.strike = 80.0 + (rng() % 40);
                t.maturity = maturity;
                t.dividend_yield = 0.01 + 0.03 * (rng() % 100) / 100.0;
                t.is_call = bool_dist(rng);
                t.equity_bucket = rng() % NUM_EQ_BUCKETS;
                portfolio.emplace_back(t);
                break;
            }
            case 2: {
                InflationSwapTrade t;
                t.notional = notional;
                t.fixed_rate = 0.02 + 0.02 * (rng() % 100) / 100.0;
                t.maturity = maturity;
                portfolio.emplace_back(t);
                break;
            }
            case 3: {
                FXOptionTrade t;
                t.notional = notional;
                t.strike = 1.0 + 0.3 * (rng() % 100) / 100.0;
                t.maturity = maturity;
                t.is_call = bool_dist(rng);
                portfolio.emplace_back(t);
                break;
            }
            case 4: {
                XCCYSwapTrade t;
                t.dom_notional = notional;
                t.fgn_notional = notional / 1.10;
                t.dom_fixed_rate = 0.03 + 0.02 * (rng() % 100) / 100.0;
                t.fgn_fixed_rate = 0.02 + 0.02 * (rng() % 100) / 100.0;
                t.maturity = maturity;
                t.frequency = 2;
                portfolio.emplace_back(t);
                break;
            }
        }
    }
    return portfolio;
}

double portfolioNPV(const std::vector<Trade>& portfolio, const MarketEnvironment& mkt,
                    int num_threads) {
    double total = 0.0;
    #pragma omp parallel for reduction(+:total) num_threads(num_threads)
    for (int i = 0; i < static_cast<int>(portfolio.size()); ++i) {
        total += priceTrade(portfolio[i], mkt);
    }
    return total;
}

int main(int argc, char* argv[]) {
    int num_trades = 100;
    int num_threads = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num-trades" && i + 1 < argc) {
            num_trades = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: isda-simm [--num-trades N] [--threads N]\n";
            return 0;
        }
    }

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    std::cout << "=== ISDA-SIMM Plain C++ Pricer ===\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n\n";

    // Build market
    auto t_start = std::chrono::high_resolution_clock::now();
    MarketEnvironment mkt = buildMarket();

    // Generate portfolio
    std::mt19937 rng(42);
    std::vector<Trade> portfolio = generatePortfolio(num_trades, rng);

    // Count trade types
    int n_irs = 0, n_eq = 0, n_inf = 0, n_fx = 0, n_xccy = 0;
    for (const auto& t : portfolio) {
        if (std::holds_alternative<IRSwapTrade>(t)) ++n_irs;
        else if (std::holds_alternative<EquityOptionTrade>(t)) ++n_eq;
        else if (std::holds_alternative<InflationSwapTrade>(t)) ++n_inf;
        else if (std::holds_alternative<FXOptionTrade>(t)) ++n_fx;
        else if (std::holds_alternative<XCCYSwapTrade>(t)) ++n_xccy;
    }
    std::cout << "Portfolio composition:\n";
    std::cout << "  IRS: " << n_irs << ", Equity Options: " << n_eq
              << ", Inflation Swaps: " << n_inf << "\n";
    std::cout << "  FX Options: " << n_fx << ", XCCY Swaps: " << n_xccy << "\n\n";

    // Compute base NPV
    double base_npv = portfolioNPV(portfolio, mkt, num_threads);
    std::cout << "Portfolio NPV: " << std::fixed << std::setprecision(2) << base_npv << "\n\n";

    // ========== Sensitivities (Bump & Revalue) ==========
    std::cout << "--- Sensitivities (Bump & Revalue) ---\n\n";

    double bp = 0.0001;  // 1 basis point for rate bumps
    double vol_bump = 0.01;  // 1% absolute vol bump

    // --- IR Delta (USD curve) ---
    std::array<double, NUM_IR_TENORS> ir_delta_usd{};
    std::cout << "IR Delta (USD):\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.usd_curve = mkt.usd_curve.bumped(k, bp);
        mkt_dn.usd_curve = mkt.usd_curve.bumped(k, -bp);
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        ir_delta_usd[k] = (npv_up - npv_dn) / (2.0 * bp);
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": "
                  << std::setw(14) << std::fixed << std::setprecision(2) << ir_delta_usd[k] << "\n";
    }

    // --- IR Delta (EUR curve) ---
    std::array<double, NUM_IR_TENORS> ir_delta_eur{};
    std::cout << "\nIR Delta (EUR):\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.eur_curve = mkt.eur_curve.bumped(k, bp);
        mkt_dn.eur_curve = mkt.eur_curve.bumped(k, -bp);
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        ir_delta_eur[k] = (npv_up - npv_dn) / (2.0 * bp);
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": "
                  << std::setw(14) << std::fixed << std::setprecision(2) << ir_delta_eur[k] << "\n";
    }

    // SIMM aggregation commented out — model computes prices and risk only
    /*
    std::array<double, NUM_IR_TENORS> ir_dv01_usd{};
    std::array<double, NUM_IR_TENORS> ir_dv01_eur{};
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        ir_dv01_usd[k] = ir_delta_usd[k] * bp;
        ir_dv01_eur[k] = ir_delta_eur[k] * bp;
    }
    SIMMResults simm_results;
    simm_results.ir_delta_margin = aggregateIRDelta(ir_dv01_usd) + aggregateIRDelta(ir_dv01_eur);
    */

    // --- Inflation Delta ---
    std::array<double, NUM_IR_TENORS> inflation_delta{};
    std::cout << "\nInflation Delta:\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.inflation = mkt.inflation.bumped(k, bp);
        mkt_dn.inflation = mkt.inflation.bumped(k, -bp);
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        inflation_delta[k] = (npv_up - npv_dn) / (2.0 * bp);
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": "
                  << std::setw(14) << std::fixed << std::setprecision(2) << inflation_delta[k] << "\n";
    }
    /*
    std::array<double, NUM_IR_TENORS> inflation_dv01{};
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        inflation_dv01[k] = inflation_delta[k] * bp;
    }
    simm_results.inflation_margin = aggregateInflation(inflation_dv01);
    */

    // --- Equity Delta ---
    double eq_delta = 0.0;
    {
        double eq_bump = mkt.equity_spot * 0.01;  // 1% of spot
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.equity_spot = mkt.equity_spot + eq_bump;
        mkt_dn.equity_spot = mkt.equity_spot - eq_bump;
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        eq_delta = (npv_up - npv_dn) / (2.0 * eq_bump);
    }
    std::cout << "\nEquity Delta: " << std::fixed << std::setprecision(2) << eq_delta << "\n";
    // simm_results.eq_delta_margin = aggregateEQDelta(eq_delta * mkt.equity_spot * 0.01);

    // --- Equity Vega ---
    double eq_vega = 0.0;
    {
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.equity_vol = mkt.equity_vol + vol_bump;
        mkt_dn.equity_vol = mkt.equity_vol - vol_bump;
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        eq_vega = (npv_up - npv_dn) / (2.0 * vol_bump);
    }
    std::cout << "Equity Vega: " << std::fixed << std::setprecision(2) << eq_vega << "\n";
    // simm_results.eq_vega_margin = std::abs(EQ_VEGA_RISK_WEIGHT * eq_vega * vol_bump);

    // --- FX Delta ---
    double fx_delta = 0.0;
    {
        double fx_bump_size = mkt.fx_spot * 0.01;  // 1% of spot
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.fx_spot = mkt.fx_spot + fx_bump_size;
        mkt_dn.fx_spot = mkt.fx_spot - fx_bump_size;
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        fx_delta = (npv_up - npv_dn) / (2.0 * fx_bump_size);
    }
    std::cout << "\nFX Delta (EURUSD): " << std::fixed << std::setprecision(2) << fx_delta << "\n";
    // std::vector<double> fx_deltas = {fx_delta * mkt.fx_spot * 0.01};
    // simm_results.fx_delta_margin = aggregateFXDelta(fx_deltas);

    // --- FX Vega ---
    double fx_vega = 0.0;
    {
        MarketEnvironment mkt_up = mkt;
        MarketEnvironment mkt_dn = mkt;
        mkt_up.fx_vol = mkt.fx_vol + vol_bump;
        mkt_dn.fx_vol = mkt.fx_vol - vol_bump;
        double npv_up = portfolioNPV(portfolio, mkt_up, num_threads);
        double npv_dn = portfolioNPV(portfolio, mkt_dn, num_threads);
        fx_vega = (npv_up - npv_dn) / (2.0 * vol_bump);
    }
    std::cout << "FX Vega (EURUSD): " << std::fixed << std::setprecision(2) << fx_vega << "\n";
    // simm_results.fx_vega_margin = std::abs(FX_VEGA_RISK_WEIGHT * fx_vega * vol_bump);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    /*
    // --- Compute total SIMM ---
    simm_results.computeTotal();

    std::cout << "\n=== SIMM Margin by Risk Class ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  IR Delta:        " << std::setw(14) << simm_results.ir_delta_margin << "\n";
    std::cout << "  IR Vega:         " << std::setw(14) << simm_results.ir_vega_margin << "\n";
    std::cout << "  Equity Delta:    " << std::setw(14) << simm_results.eq_delta_margin << "\n";
    std::cout << "  Equity Vega:     " << std::setw(14) << simm_results.eq_vega_margin << "\n";
    std::cout << "  FX Delta:        " << std::setw(14) << simm_results.fx_delta_margin << "\n";
    std::cout << "  FX Vega:         " << std::setw(14) << simm_results.fx_vega_margin << "\n";
    std::cout << "  Inflation:       " << std::setw(14) << simm_results.inflation_margin << "\n";
    std::cout << "  TOTAL SIMM:      " << std::setw(14) << simm_results.total_margin << "\n";
    */

    std::cout << "\nComputation time: " << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n";

    return 0;
}
