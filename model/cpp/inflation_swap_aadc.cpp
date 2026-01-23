// Inflation Swap Pricer with AADC
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
// AADC Kernel Structure for Inflation Swap
// ============================================================================
struct InflKernelArgs {
    std::array<AADCArgument, NUM_IR_TENORS> ir_curve_args;    // diff
    std::array<AADCArgument, NUM_IR_TENORS> infl_rate_args;   // diff
    AADCArgument notional_arg;      // no-diff
    AADCArgument fixed_rate_arg;    // no-diff
    AADCArgument base_cpi_arg;      // no-diff
    AADCResult npv_res;
};

// Record inflation swap kernel: 5y zero-coupon structure
void recordInflKernel(AADCFunctions<mmType>& funcs, InflKernelArgs& args) {
    funcs.startRecording();

    // IR curve rates - differentiable
    std::array<idouble, NUM_IR_TENORS> ir_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        ir_rates[i] = idouble(0.04);
        args.ir_curve_args[i] = AADCArgument(ir_rates[i].markAsInput());
    }

    // Inflation rates - differentiable
    std::array<idouble, NUM_IR_TENORS> infl_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        infl_rates[i] = idouble(0.025);
        args.infl_rate_args[i] = AADCArgument(infl_rates[i].markAsInput());
    }

    // Trade parameters - non-differentiable
    idouble notional(1e6);
    idouble fixed_rate(0.025);
    idouble base_cpi(100.0);
    args.notional_arg = AADCArgument(notional.markAsInputNoDiff());
    args.fixed_rate_arg = AADCArgument(fixed_rate.markAsInputNoDiff());
    args.base_cpi_arg = AADCArgument(base_cpi.markAsInputNoDiff());

    // Fixed structure: 5y zero-coupon inflation swap
    constexpr double tau = 5.0;

    // Interpolate IR rate at maturity
    idouble r_t = ir_rates[0];
    for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
        if (tau >= IR_TENORS[k] && tau <= IR_TENORS[k + 1]) {
            double w = (tau - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
            r_t = ir_rates[k] * (1.0 - w) + ir_rates[k + 1] * w;
            break;
        }
    }
    idouble df = std::exp(-r_t * tau);

    // Interpolate inflation rate at maturity
    idouble infl_t = infl_rates[0];
    for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
        if (tau >= IR_TENORS[k] && tau <= IR_TENORS[k + 1]) {
            double w = (tau - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
            infl_t = infl_rates[k] * (1.0 - w) + infl_rates[k + 1] * w;
            break;
        }
    }

    // Fixed leg: N * (exp(K*T) - 1) * DF
    idouble fixed_leg = notional * (std::exp(fixed_rate * tau) - 1.0) * df;

    // Inflation leg: N * (CPI_T/CPI_0 - 1) * DF
    idouble cpi_T = base_cpi * std::exp(infl_t * tau);
    idouble cpi_ratio = cpi_T / base_cpi;
    idouble infl_leg = notional * (cpi_ratio - 1.0) * df;

    // NPV from inflation receiver perspective
    idouble npv = infl_leg - fixed_leg;
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

    std::cout << "=== Inflation Swap AADC Pricer ===\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n\n";

    // Record kernel
    auto rec_start = std::chrono::high_resolution_clock::now();
    AADCFunctions<mmType> funcs;
    InflKernelArgs args;
    recordInflKernel(funcs, args);
    auto rec_end = std::chrono::high_resolution_clock::now();
    double recording_time = std::chrono::duration<double>(rec_end - rec_start).count();
    double kernel_memory_mb = static_cast<double>(funcs.getMemUse() + funcs.getWorkSpaceMemUse()) / (1024.0 * 1024.0);

    std::cout << "Kernel recorded in " << std::fixed << std::setprecision(3)
              << recording_time * 1000 << " ms\n\n";

    // Generate trades
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> not_dist(1e6, 50e6);
    std::uniform_real_distribution<double> rate_dist(0.02, 0.04);

    struct TradeParams { double notional; double fixed_rate; };
    std::vector<TradeParams> trades(num_trades);
    for (int i = 0; i < num_trades; ++i) {
        trades[i] = {not_dist(rng), rate_dist(rng)};
    }

    // Market data
    std::array<double, NUM_IR_TENORS> usd_rates, infl_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        usd_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
        infl_rates[i] = 0.025 + 0.002 * IR_TENORS[i] / 30.0;
    }

    // Evaluate
    auto eval_start = std::chrono::high_resolution_clock::now();

    std::vector<double> prices(num_trades);
    std::array<double, NUM_IR_TENORS> ir_delta_sum{};
    std::array<double, NUM_IR_TENORS> infl_delta_sum{};

    auto ws = std::shared_ptr<AADCWorkSpace<mmType>>(funcs.createWorkSpace());
    int num_batches = (num_trades + AVX_WIDTH - 1) / AVX_WIDTH;

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * AVX_WIDTH;

        // Set market data
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            ws->setVal(args.ir_curve_args[k], mmSetConst<mmType>(usd_rates[k]));
            ws->setVal(args.infl_rate_args[k], mmSetConst<mmType>(infl_rates[k]));
        }
        ws->setVal(args.base_cpi_arg, mmSetConst<mmType>(100.0));

        // Per-trade params
        mmType mm_not, mm_rate;
        double* n_ptr = reinterpret_cast<double*>(&mm_not);
        double* r_ptr = reinterpret_cast<double*>(&mm_rate);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) {
                n_ptr[lane] = trades[idx].notional;
                r_ptr[lane] = trades[idx].fixed_rate;
            } else {
                n_ptr[lane] = 0.0;
                r_ptr[lane] = 0.025;
            }
        }
        ws->setVal(args.notional_arg, mm_not);
        ws->setVal(args.fixed_rate_arg, mm_rate);

        // Forward
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

        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            ir_delta_sum[k] += mmSum(ws->diff(args.ir_curve_args[k]));
            infl_delta_sum[k] += mmSum(ws->diff(args.infl_rate_args[k]));
        }
    }

    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();

    double portfolio_npv = 0.0;
    for (double p : prices) portfolio_npv += p;

    std::cout << "Portfolio NPV: " << std::fixed << std::setprecision(2) << portfolio_npv << "\n\n";

    std::cout << "IR Delta (AAD):\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k)
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": " << std::setw(14) << ir_delta_sum[k] << "\n";

    std::cout << "\nInflation Delta (AAD):\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k)
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": " << std::setw(14) << infl_delta_sum[k] << "\n";

    // SIMM aggregation commented out — models compute prices and risk only
    /*
    std::array<double, NUM_IR_TENORS> ir_dv01{}, infl_dv01{};
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        ir_dv01[k] = ir_delta_sum[k] * 0.0001;
        infl_dv01[k] = infl_delta_sum[k] * 0.0001;
    }
    double ir_delta_margin = aggregateIRDelta(ir_dv01);
    double inflation_margin = aggregateInflation(infl_dv01);

    SIMMResults simm_results;
    simm_results.ir_delta_margin = ir_delta_margin;
    simm_results.inflation_margin = inflation_margin;
    simm_results.computeTotal();

    std::cout << "\n=== SIMM Margin ===\n";
    std::cout << "  IR Delta:    " << std::setw(14) << ir_delta_margin << "\n";
    std::cout << "  Inflation:   " << std::setw(14) << inflation_margin << "\n";
    std::cout << "  TOTAL SIMM:  " << std::setw(14) << simm_results.total_margin << "\n\n";

    SIMMExecutionLogger logger("data/execution_log.csv");
    logger.log("inflation_swap_aadc_cpp", MODEL_VERSION, "price_with_greeks",
               num_trades, num_threads, portfolio_npv, simm_results.total_margin,
               ir_delta_margin, 0.0, 0.0, 0.0, 0.0, inflation_margin,
               eval_time, recording_time, kernel_memory_mb,
               "C++", true, AVX_WIDTH);
    */

    std::cout << "Eval time: " << std::setprecision(3) << eval_time * 1000 << " ms\n";

    return 0;
}
