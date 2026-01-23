// FX Option (Garman-Kohlhagen) Pricer with AADC
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
// AADC Kernel Structure for FX Option
// ============================================================================
struct FXKernelArgs {
    AADCArgument spot_arg;          // diff
    AADCArgument vol_arg;           // diff
    std::array<AADCArgument, NUM_IR_TENORS> dom_curve_args;  // diff
    std::array<AADCArgument, NUM_IR_TENORS> fgn_curve_args;  // diff
    AADCArgument strike_arg;        // no-diff
    AADCResult price_res;
};

// Record FX option kernel: 1y EURUSD call
void recordFXKernel(AADCFunctions<mmType>& funcs, FXKernelArgs& args) {
    funcs.startRecording();

    // Market data - differentiable
    idouble spot(1.10);
    idouble vol(0.12);
    args.spot_arg = AADCArgument(spot.markAsInput());
    args.vol_arg = AADCArgument(vol.markAsInput());

    std::array<idouble, NUM_IR_TENORS> dom_rates, fgn_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        dom_rates[i] = idouble(0.04);
        fgn_rates[i] = idouble(0.03);
        args.dom_curve_args[i] = AADCArgument(dom_rates[i].markAsInput());
        args.fgn_curve_args[i] = AADCArgument(fgn_rates[i].markAsInput());
    }

    // Trade parameter
    idouble strike(1.10);
    args.strike_arg = AADCArgument(strike.markAsInputNoDiff());

    // Fixed structure: 1y FX call
    constexpr double tau = 1.0;

    // Interpolate domestic rate
    idouble rd = dom_rates[0];
    for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
        if (tau >= IR_TENORS[k] && tau <= IR_TENORS[k + 1]) {
            double w = (tau - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
            rd = dom_rates[k] * (1.0 - w) + dom_rates[k + 1] * w;
            break;
        }
    }

    // Interpolate foreign rate
    idouble rf = fgn_rates[0];
    for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
        if (tau >= IR_TENORS[k] && tau <= IR_TENORS[k + 1]) {
            double w = (tau - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
            rf = fgn_rates[k] * (1.0 - w) + fgn_rates[k + 1] * w;
            break;
        }
    }

    idouble sqrt_tau = std::sqrt(idouble(tau));
    idouble d1 = (std::log(spot / strike) + (rd - rf + 0.5 * vol * vol) * tau) /
                 (vol * sqrt_tau);
    idouble d2 = d1 - vol * sqrt_tau;

    idouble df_dom = std::exp(-rd * tau);
    idouble df_fgn = std::exp(-rf * tau);

    // CDF via std::erf (AADC provides overload in namespace std)
    idouble sqrt2 = std::sqrt(idouble(2.0));
    idouble nd1 = 0.5 * (1.0 + std::erf(d1 / sqrt2));
    idouble nd2 = 0.5 * (1.0 + std::erf(d2 / sqrt2));

    // Call on foreign currency (1 unit notional in foreign)
    idouble price = spot * df_fgn * nd1 - strike * df_dom * nd2;

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

    std::cout << "=== FX Option AADC Pricer ===\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n\n";

    // Record kernel
    auto rec_start = std::chrono::high_resolution_clock::now();
    AADCFunctions<mmType> funcs;
    FXKernelArgs args;
    recordFXKernel(funcs, args);
    auto rec_end = std::chrono::high_resolution_clock::now();
    double recording_time = std::chrono::duration<double>(rec_end - rec_start).count();
    double kernel_memory_mb = static_cast<double>(funcs.getMemUse() + funcs.getWorkSpaceMemUse()) / (1024.0 * 1024.0);

    std::cout << "Kernel recorded in " << std::fixed << std::setprecision(3)
              << recording_time * 1000 << " ms\n\n";

    // Generate trades: varying strike
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> strike_dist(1.0, 1.25);
    std::vector<double> strikes(num_trades);
    for (int i = 0; i < num_trades; ++i) strikes[i] = strike_dist(rng);

    // Market data
    double spot = 1.10, vol = 0.12;
    std::array<double, NUM_IR_TENORS> dom_rates, fgn_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        dom_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
        fgn_rates[i] = 0.03 + 0.005 * IR_TENORS[i] / 30.0;
    }

    // Evaluate
    auto eval_start = std::chrono::high_resolution_clock::now();

    std::vector<double> prices(num_trades);
    double fx_delta_sum = 0.0, fx_vega_sum = 0.0;
    std::array<double, NUM_IR_TENORS> dom_delta_sum{}, fgn_delta_sum{};

    auto ws = std::shared_ptr<AADCWorkSpace<mmType>>(funcs.createWorkSpace());
    int num_batches = (num_trades + AVX_WIDTH - 1) / AVX_WIDTH;

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * AVX_WIDTH;

        ws->setVal(args.spot_arg, mmSetConst<mmType>(spot));
        ws->setVal(args.vol_arg, mmSetConst<mmType>(vol));
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            ws->setVal(args.dom_curve_args[k], mmSetConst<mmType>(dom_rates[k]));
            ws->setVal(args.fgn_curve_args[k], mmSetConst<mmType>(fgn_rates[k]));
        }

        mmType mm_strike;
        double* s_ptr = reinterpret_cast<double*>(&mm_strike);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            s_ptr[lane] = (idx < num_trades) ? strikes[idx] : 1.10;
        }
        ws->setVal(args.strike_arg, mm_strike);

        funcs.forward(*ws);

        mmType mm_price = ws->val(args.price_res);
        double* p_ptr = reinterpret_cast<double*>(&mm_price);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) prices[idx] = p_ptr[lane];
        }

        // Reverse
        ws->resetDiff();
        ws->setDiff(args.price_res, 1.0);
        funcs.reverse(*ws);

        // Accumulate sensitivities
        double* fd = reinterpret_cast<double*>(&ws->diff(args.spot_arg));
        double* fv = reinterpret_cast<double*>(&ws->diff(args.vol_arg));
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) {
                fx_delta_sum += fd[lane];
                fx_vega_sum += fv[lane];
            }
        }
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            dom_delta_sum[k] += mmSum(ws->diff(args.dom_curve_args[k]));
            fgn_delta_sum[k] += mmSum(ws->diff(args.fgn_curve_args[k]));
        }
    }

    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();

    double portfolio_npv = 0.0;
    for (double p : prices) portfolio_npv += p;

    std::cout << "Portfolio NPV: " << std::fixed << std::setprecision(2) << portfolio_npv << "\n";
    std::cout << "FX Delta (summed): " << fx_delta_sum << "\n";
    std::cout << "FX Vega (summed):  " << fx_vega_sum << "\n\n";

    // SIMM aggregation commented out — models compute prices and risk only
    /*
    double fx_delta_sens = fx_delta_sum * spot * 0.01;
    std::vector<double> fx_deltas_vec = {fx_delta_sens};
    double fx_delta_margin = aggregateFXDelta(fx_deltas_vec);
    double fx_vega_margin = std::abs(FX_VEGA_RISK_WEIGHT * fx_vega_sum * 0.01);

    std::array<double, NUM_IR_TENORS> dom_dv01{}, fgn_dv01{};
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        dom_dv01[k] = dom_delta_sum[k] * 0.0001;
        fgn_dv01[k] = fgn_delta_sum[k] * 0.0001;
    }
    double ir_delta_margin = aggregateIRDelta(dom_dv01) + aggregateIRDelta(fgn_dv01);

    SIMMResults simm_results;
    simm_results.ir_delta_margin = ir_delta_margin;
    simm_results.fx_delta_margin = fx_delta_margin;
    simm_results.fx_vega_margin = fx_vega_margin;
    simm_results.computeTotal();

    std::cout << "=== SIMM Margin ===\n";
    std::cout << "  IR Delta:    " << std::setw(14) << ir_delta_margin << "\n";
    std::cout << "  FX Delta:    " << std::setw(14) << fx_delta_margin << "\n";
    std::cout << "  FX Vega:     " << std::setw(14) << fx_vega_margin << "\n";
    std::cout << "  TOTAL SIMM:  " << std::setw(14) << simm_results.total_margin << "\n\n";

    SIMMExecutionLogger logger("data/execution_log.csv");
    logger.log("fx_option_aadc_cpp", MODEL_VERSION, "price_with_greeks",
               num_trades, num_threads, portfolio_npv, simm_results.total_margin,
               ir_delta_margin, 0.0, 0.0, fx_delta_margin, fx_vega_margin, 0.0,
               eval_time, recording_time, kernel_memory_mb,
               "C++", true, AVX_WIDTH);
    */

    std::cout << "Eval time: " << std::setprecision(3) << eval_time * 1000 << " ms\n";

    return 0;
}
