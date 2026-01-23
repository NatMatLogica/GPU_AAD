// Equity Option (Black-Scholes) Pricer with AADC
// Version: 1.0.0
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <memory>

#include <aadc/aadc.h>
#include <aadc/aadc_matrix.h>

#include "simm_config.h"
#include "market_data.h"
#include "execution_logger.h"

using namespace aadc;
using namespace simm;

typedef __m256d mmType;
constexpr int AVX_WIDTH = 4;
static const char* MODEL_VERSION = "1.0.0";

// ============================================================================
// AADC Kernel Structure for Equity Option
// ============================================================================
struct EQKernelArgs {
    AADCArgument spot_arg;         // differentiable
    AADCArgument vol_arg;          // differentiable
    std::array<AADCArgument, NUM_IR_TENORS> curve_args;  // differentiable
    AADCArgument strike_arg;       // non-diff (varies per trade)
    AADCArgument div_yield_arg;    // non-diff
    AADCResult price_res;
};

// Record equity option kernel: 1y ATM call, fixed structure
void recordEQKernel(AADCFunctions<mmType>& funcs, EQKernelArgs& args) {
    funcs.startRecording();

    // Market data - differentiable
    idouble spot(100.0);
    idouble vol(0.25);
    args.spot_arg = AADCArgument(spot.markAsInput());
    args.vol_arg = AADCArgument(vol.markAsInput());

    std::array<idouble, NUM_IR_TENORS> rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        rates[i] = idouble(0.04);
        args.curve_args[i] = AADCArgument(rates[i].markAsInput());
    }

    // Trade parameters - non-differentiable
    idouble strike(100.0);
    idouble div_yield(0.02);
    args.strike_arg = AADCArgument(strike.markAsInputNoDiff());
    args.div_yield_arg = AADCArgument(div_yield.markAsInputNoDiff());

    // Fixed structure: 1y call option, notional = strike (1 contract equivalent)
    constexpr double tau = 1.0;

    // Interpolate rate at maturity
    idouble r = rates[0];  // Will be overwritten
    for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
        if (tau >= IR_TENORS[k] && tau <= IR_TENORS[k + 1]) {
            double w = (tau - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
            r = rates[k] * (1.0 - w) + rates[k + 1] * w;
            break;
        }
    }

    idouble sqrt_tau = std::sqrt(idouble(tau));
    idouble d1 = (std::log(spot / strike) + (r - div_yield + 0.5 * vol * vol) * tau) /
                 (vol * sqrt_tau);
    idouble d2 = d1 - vol * sqrt_tau;

    idouble df = std::exp(-r * tau);
    idouble dq = std::exp(-div_yield * tau);

    // Call price (1 contract)
    idouble sqrt2 = std::sqrt(idouble(2.0));
    idouble nd1 = 0.5 * (1.0 + std::erf(d1 / sqrt2));
    idouble nd2 = 0.5 * (1.0 + std::erf(d2 / sqrt2));
    idouble price = spot * dq * nd1 - strike * df * nd2;

    args.price_res = AADCResult(price.markAsOutput());
    funcs.stopRecording();
}

int main(int argc, char* argv[]) {
    int num_trades = 100;
    int num_threads = 4;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num-trades" && i + 1 < argc) num_trades = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::stoi(argv[++i]);
    }

    std::cout << "=== Equity Option AADC Pricer ===\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n\n";

    // Record kernel
    auto rec_start = std::chrono::high_resolution_clock::now();
    AADCFunctions<mmType> funcs;
    EQKernelArgs args;
    recordEQKernel(funcs, args);
    auto rec_end = std::chrono::high_resolution_clock::now();
    double recording_time = std::chrono::duration<double>(rec_end - rec_start).count();
    double kernel_memory_mb = static_cast<double>(funcs.getMemUse() + funcs.getWorkSpaceMemUse()) / (1024.0 * 1024.0);

    std::cout << "Kernel recorded in " << std::fixed << std::setprecision(3)
              << recording_time * 1000 << " ms, memory: " << kernel_memory_mb << " MB\n\n";

    // Generate trades: varying strike and div yield
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> strike_dist(80.0, 120.0);
    std::uniform_real_distribution<double> div_dist(0.01, 0.04);

    struct TradeParams { double strike; double div_yield; };
    std::vector<TradeParams> trades(num_trades);
    for (int i = 0; i < num_trades; ++i) {
        trades[i] = {strike_dist(rng), div_dist(rng)};
    }

    // Market data
    double spot = 100.0;
    double vol = 0.25;
    std::array<double, NUM_IR_TENORS> usd_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        usd_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
    }

    // Evaluate
    auto eval_start = std::chrono::high_resolution_clock::now();

    std::vector<double> prices(num_trades);
    double delta_sum = 0.0, vega_sum = 0.0;

    auto ws = std::shared_ptr<AADCWorkSpace<mmType>>(funcs.createWorkSpace());
    int num_batches = (num_trades + AVX_WIDTH - 1) / AVX_WIDTH;

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * AVX_WIDTH;

        // Set market data (same for all trades)
        ws->setVal(args.spot_arg, mmSetConst<mmType>(spot));
        ws->setVal(args.vol_arg, mmSetConst<mmType>(vol));
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            ws->setVal(args.curve_args[k], mmSetConst<mmType>(usd_rates[k]));
        }

        // Set per-trade parameters
        mmType mm_strike, mm_div;
        double* str_ptr = reinterpret_cast<double*>(&mm_strike);
        double* div_ptr = reinterpret_cast<double*>(&mm_div);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) {
                str_ptr[lane] = trades[idx].strike;
                div_ptr[lane] = trades[idx].div_yield;
            } else {
                str_ptr[lane] = 100.0;
                div_ptr[lane] = 0.02;
            }
        }
        ws->setVal(args.strike_arg, mm_strike);
        ws->setVal(args.div_yield_arg, mm_div);

        // Forward
        funcs.forward(*ws);

        mmType mm_price = ws->val(args.price_res);
        double* p_ptr = reinterpret_cast<double*>(&mm_price);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) prices[idx] = p_ptr[lane];
        }

        // Reverse for Greeks
        ws->resetDiff();
        ws->setDiff(args.price_res, 1.0);
        funcs.reverse(*ws);

        mmType mm_delta = ws->diff(args.spot_arg);
        mmType mm_vega = ws->diff(args.vol_arg);

        // Only accumulate valid lanes
        double* d_ptr = reinterpret_cast<double*>(&mm_delta);
        double* v_ptr = reinterpret_cast<double*>(&mm_vega);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) {
                delta_sum += d_ptr[lane];
                vega_sum += v_ptr[lane];
            }
        }
    }

    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();

    double portfolio_npv = 0.0;
    for (double p : prices) portfolio_npv += p;

    std::cout << "Portfolio NPV: " << std::fixed << std::setprecision(2) << portfolio_npv << "\n\n";
    std::cout << "Equity Delta (summed): " << delta_sum << "\n";
    std::cout << "Equity Vega (summed):  " << vega_sum << "\n\n";

    // SIMM aggregation commented out — models compute prices and risk only
    /*
    double eq_delta_sensitivity = delta_sum * spot * 0.01;
    double eq_delta_margin = aggregateEQDelta(eq_delta_sensitivity);
    double eq_vega_margin = std::abs(EQ_VEGA_RISK_WEIGHT * vega_sum * 0.01);

    SIMMResults simm_results;
    simm_results.eq_delta_margin = eq_delta_margin;
    simm_results.eq_vega_margin = eq_vega_margin;
    simm_results.computeTotal();

    std::cout << "=== SIMM Margin ===\n";
    std::cout << "  Equity Delta: " << std::setw(14) << eq_delta_margin << "\n";
    std::cout << "  Equity Vega:  " << std::setw(14) << eq_vega_margin << "\n";
    std::cout << "  TOTAL SIMM:   " << std::setw(14) << simm_results.total_margin << "\n\n";

    SIMMExecutionLogger logger("data/execution_log.csv");
    logger.log("equity_option_aadc_cpp", MODEL_VERSION, "price_with_greeks",
               num_trades, num_threads, portfolio_npv, simm_results.total_margin,
               0.0, eq_delta_margin, eq_vega_margin, 0.0, 0.0, 0.0,
               eval_time, recording_time, kernel_memory_mb,
               "C++", true, AVX_WIDTH);
    */

    std::cout << "Eval time: " << std::setprecision(3) << eval_time * 1000 << " ms\n";
    std::cout << "Recording time: " << recording_time * 1000 << " ms\n";

    return 0;
}
