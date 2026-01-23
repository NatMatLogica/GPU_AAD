// Cross-Currency Swap Pricer with AADC
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
// AADC Kernel Structure for XCCY Swap
// ============================================================================
struct XCCYKernelArgs {
    AADCArgument fx_spot_arg;       // diff
    std::array<AADCArgument, NUM_IR_TENORS> dom_curve_args;  // diff
    std::array<AADCArgument, NUM_IR_TENORS> fgn_curve_args;  // diff
    AADCArgument dom_notional_arg;  // no-diff
    AADCArgument fgn_notional_arg;  // no-diff
    AADCArgument dom_rate_arg;      // no-diff
    AADCArgument fgn_rate_arg;      // no-diff
    AADCResult npv_res;
};

// Record XCCY swap kernel: 5y semi-annual with notional exchange
void recordXCCYKernel(AADCFunctions<mmType>& funcs, XCCYKernelArgs& args) {
    funcs.startRecording();

    // FX spot - differentiable
    idouble fx_spot(1.10);
    args.fx_spot_arg = AADCArgument(fx_spot.markAsInput());

    // Curves - differentiable
    std::array<idouble, NUM_IR_TENORS> dom_rates, fgn_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        dom_rates[i] = idouble(0.04);
        fgn_rates[i] = idouble(0.03);
        args.dom_curve_args[i] = AADCArgument(dom_rates[i].markAsInput());
        args.fgn_curve_args[i] = AADCArgument(fgn_rates[i].markAsInput());
    }

    // Trade params - non-diff
    idouble dom_notional(1e6);
    idouble fgn_notional(909090.9);
    idouble dom_rate(0.04);
    idouble fgn_rate(0.03);
    args.dom_notional_arg = AADCArgument(dom_notional.markAsInputNoDiff());
    args.fgn_notional_arg = AADCArgument(fgn_notional.markAsInputNoDiff());
    args.dom_rate_arg = AADCArgument(dom_rate.markAsInputNoDiff());
    args.fgn_rate_arg = AADCArgument(fgn_rate.markAsInputNoDiff());

    // Fixed structure: 5y semi-annual with notional exchange
    constexpr double maturity = 5.0;
    constexpr int frequency = 2;
    constexpr double dt = 1.0 / frequency;
    constexpr int num_periods = static_cast<int>(maturity * frequency);

    // Helper: interpolate rate from curve at time t
    auto interpRate = [&](const std::array<idouble, NUM_IR_TENORS>& rates, double t) -> idouble {
        if (t <= IR_TENORS[0]) return rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return rates[NUM_IR_TENORS - 1];
        for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
            if (t >= IR_TENORS[k] && t <= IR_TENORS[k + 1]) {
                double w = (t - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
                return rates[k] * (1.0 - w) + rates[k + 1] * w;
            }
        }
        return rates[NUM_IR_TENORS - 1];
    };

    // Domestic leg
    idouble dom_leg(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        idouble r = interpRate(dom_rates, t);
        idouble df = std::exp(-r * t);
        dom_leg = dom_leg + dom_notional * dom_rate * dt * df;
    }

    // Foreign leg
    idouble fgn_leg(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        idouble r = interpRate(fgn_rates, t);
        idouble df = std::exp(-r * t);
        fgn_leg = fgn_leg + fgn_notional * fgn_rate * dt * df;
    }

    // Notional exchange at maturity
    idouble dom_r_mat = interpRate(dom_rates, maturity);
    idouble fgn_r_mat = interpRate(fgn_rates, maturity);
    idouble dom_df_mat = std::exp(-dom_r_mat * maturity);
    idouble fgn_df_mat = std::exp(-fgn_r_mat * maturity);

    dom_leg = dom_leg + dom_notional * dom_df_mat;
    fgn_leg = fgn_leg + fgn_notional * fgn_df_mat;

    // Inception exchange (t=0)
    dom_leg = dom_leg - dom_notional;
    fgn_leg = fgn_leg - fgn_notional;

    // NPV in domestic currency
    idouble npv = dom_leg - fgn_leg * fx_spot;
    args.npv_res = AADCResult(npv.markAsOutput());

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

    std::cout << "=== XCCY Swap AADC Pricer ===\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n\n";

    // Record kernel
    auto rec_start = std::chrono::high_resolution_clock::now();
    AADCFunctions<mmType> funcs;
    XCCYKernelArgs args;
    recordXCCYKernel(funcs, args);
    auto rec_end = std::chrono::high_resolution_clock::now();
    double recording_time = std::chrono::duration<double>(rec_end - rec_start).count();
    double kernel_memory_mb = static_cast<double>(funcs.getMemUse() + funcs.getWorkSpaceMemUse()) / (1024.0 * 1024.0);

    std::cout << "Kernel recorded in " << std::fixed << std::setprecision(3)
              << recording_time * 1000 << " ms\n\n";

    // Generate trades
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> not_dist(1e6, 50e6);
    std::uniform_real_distribution<double> dom_rate_dist(0.03, 0.05);
    std::uniform_real_distribution<double> fgn_rate_dist(0.02, 0.04);

    struct TradeParams { double dom_not; double fgn_not; double dom_rate; double fgn_rate; };
    std::vector<TradeParams> trades(num_trades);
    double fx_spot_init = 1.10;
    for (int i = 0; i < num_trades; ++i) {
        double dn = not_dist(rng);
        trades[i] = {dn, dn / fx_spot_init, dom_rate_dist(rng), fgn_rate_dist(rng)};
    }

    // Market data
    double fx_spot = 1.10;
    std::array<double, NUM_IR_TENORS> dom_rates, fgn_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        dom_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
        fgn_rates[i] = 0.03 + 0.005 * IR_TENORS[i] / 30.0;
    }

    // Evaluate
    auto eval_start = std::chrono::high_resolution_clock::now();

    std::vector<double> prices(num_trades);
    double fx_delta_sum = 0.0;
    std::array<double, NUM_IR_TENORS> dom_delta_sum{}, fgn_delta_sum{};

    auto ws = std::shared_ptr<AADCWorkSpace<mmType>>(funcs.createWorkSpace());
    int num_batches = (num_trades + AVX_WIDTH - 1) / AVX_WIDTH;

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * AVX_WIDTH;

        ws->setVal(args.fx_spot_arg, mmSetConst<mmType>(fx_spot));
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            ws->setVal(args.dom_curve_args[k], mmSetConst<mmType>(dom_rates[k]));
            ws->setVal(args.fgn_curve_args[k], mmSetConst<mmType>(fgn_rates[k]));
        }

        // Per-trade params
        mmType mm_dn, mm_fn, mm_dr, mm_fr;
        double* dn_p = reinterpret_cast<double*>(&mm_dn);
        double* fn_p = reinterpret_cast<double*>(&mm_fn);
        double* dr_p = reinterpret_cast<double*>(&mm_dr);
        double* fr_p = reinterpret_cast<double*>(&mm_fr);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) {
                dn_p[lane] = trades[idx].dom_not;
                fn_p[lane] = trades[idx].fgn_not;
                dr_p[lane] = trades[idx].dom_rate;
                fr_p[lane] = trades[idx].fgn_rate;
            } else {
                dn_p[lane] = fn_p[lane] = 0.0;
                dr_p[lane] = 0.04;
                fr_p[lane] = 0.03;
            }
        }
        ws->setVal(args.dom_notional_arg, mm_dn);
        ws->setVal(args.fgn_notional_arg, mm_fn);
        ws->setVal(args.dom_rate_arg, mm_dr);
        ws->setVal(args.fgn_rate_arg, mm_fr);

        funcs.forward(*ws);

        mmType mm_npv = ws->val(args.npv_res);
        double* p_ptr = reinterpret_cast<double*>(&mm_npv);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) prices[idx] = p_ptr[lane];
        }

        // Reverse
        ws->resetDiff();
        ws->setDiff(args.npv_res, 1.0);
        funcs.reverse(*ws);

        // FX delta
        double* fd = reinterpret_cast<double*>(&ws->diff(args.fx_spot_arg));
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) fx_delta_sum += fd[lane];
        }

        // IR deltas
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
    std::cout << "FX Delta (summed): " << fx_delta_sum << "\n\n";

    std::cout << "IR Delta DOM (AAD):\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k)
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": " << std::setw(14) << dom_delta_sum[k] << "\n";

    // SIMM aggregation commented out — models compute prices and risk only
    /*
    double fx_delta_sens = fx_delta_sum * fx_spot * 0.01;
    std::vector<double> fx_deltas_vec = {fx_delta_sens};
    double fx_delta_margin = aggregateFXDelta(fx_deltas_vec);

    std::array<double, NUM_IR_TENORS> dom_dv01{}, fgn_dv01{};
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        dom_dv01[k] = dom_delta_sum[k] * 0.0001;
        fgn_dv01[k] = fgn_delta_sum[k] * 0.0001;
    }
    double ir_delta_margin = aggregateIRDelta(dom_dv01) + aggregateIRDelta(fgn_dv01);

    SIMMResults simm_results;
    simm_results.ir_delta_margin = ir_delta_margin;
    simm_results.fx_delta_margin = fx_delta_margin;
    simm_results.computeTotal();

    std::cout << "\n=== SIMM Margin ===\n";
    std::cout << "  IR Delta:    " << std::setw(14) << ir_delta_margin << "\n";
    std::cout << "  FX Delta:    " << std::setw(14) << fx_delta_margin << "\n";
    std::cout << "  TOTAL SIMM:  " << std::setw(14) << simm_results.total_margin << "\n\n";

    SIMMExecutionLogger logger("data/execution_log.csv");
    logger.log("xccy_swap_aadc_cpp", MODEL_VERSION, "price_with_greeks",
               num_trades, num_threads, portfolio_npv, simm_results.total_margin,
               ir_delta_margin, 0.0, 0.0, fx_delta_margin, 0.0, 0.0,
               eval_time, recording_time, kernel_memory_mb,
               "C++", true, AVX_WIDTH);
    */

    std::cout << "Eval time: " << std::setprecision(3) << eval_time * 1000 << " ms\n";

    return 0;
}
