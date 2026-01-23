// Vanilla IRS Pricer with AADC - Scalar recording, kernel reuse for portfolio
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
// AADC Kernel Structure for IRS
// ============================================================================
struct IRSKernelArgs {
    std::array<AADCArgument, NUM_IR_TENORS> curve_args;  // differentiable
    AADCArgument notional_arg;    // non-diff (varies per trade)
    AADCArgument fixed_rate_arg;  // non-diff (varies per trade)
    AADCResult npv_res;
};

// Record IRS kernel: 5y semi-annual (10 periods), fixed structure
void recordIRSKernel(AADCFunctions<mmType>& funcs, IRSKernelArgs& args) {
    funcs.startRecording();

    // Yield curve rates - differentiable
    std::array<idouble, NUM_IR_TENORS> rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        rates[i] = idouble(0.04);
        args.curve_args[i] = AADCArgument(rates[i].markAsInput());
    }

    // Trade parameters - non-differentiable
    idouble notional(1e6);
    idouble fixed_rate(0.03);
    args.notional_arg = AADCArgument(notional.markAsInputNoDiff());
    args.fixed_rate_arg = AADCArgument(fixed_rate.markAsInputNoDiff());

    // Build yield curve with idouble rates
    // Fixed structure: 5y maturity, semi-annual (10 periods)
    constexpr double maturity = 5.0;
    constexpr int frequency = 2;
    constexpr double dt = 1.0 / frequency;
    constexpr int num_periods = static_cast<int>(maturity * frequency);

    // Pricing math: same as priceVanillaIRS but with explicit idouble
    idouble fixed_leg(0.0);
    idouble floating_leg(0.0);

    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;

        // Interpolate zero rate at time t
        idouble r_t;
        if (t <= IR_TENORS[0]) {
            r_t = rates[0];
        } else if (t >= IR_TENORS[NUM_IR_TENORS - 1]) {
            r_t = rates[NUM_IR_TENORS - 1];
        } else {
            r_t = rates[0]; // default
            for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
                if (t >= IR_TENORS[k] && t <= IR_TENORS[k + 1]) {
                    double w = (t - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
                    r_t = rates[k] * (1.0 - w) + rates[k + 1] * w;
                    break;
                }
            }
        }

        // Discount factor
        idouble df = std::exp(-r_t * t);

        // Fixed leg
        fixed_leg = fixed_leg + notional * fixed_rate * dt * df;

        // Floating leg: forward rate
        double t_prev = (i - 1) * dt;
        idouble r_prev;
        if (t_prev <= IR_TENORS[0] || t_prev <= 0.0) {
            r_prev = rates[0];
        } else if (t_prev >= IR_TENORS[NUM_IR_TENORS - 1]) {
            r_prev = rates[NUM_IR_TENORS - 1];
        } else {
            r_prev = rates[0];
            for (int k = 0; k < NUM_IR_TENORS - 1; ++k) {
                if (t_prev >= IR_TENORS[k] && t_prev <= IR_TENORS[k + 1]) {
                    double w = (t_prev - IR_TENORS[k]) / (IR_TENORS[k + 1] - IR_TENORS[k]);
                    r_prev = rates[k] * (1.0 - w) + rates[k + 1] * w;
                    break;
                }
            }
        }
        idouble df_prev = std::exp(-r_prev * t_prev);
        idouble fwd = std::log(df_prev / df) / (t - t_prev + 1e-15);
        floating_leg = floating_leg + notional * dt * fwd * df;
    }

    // NPV: pay fixed, receive floating
    idouble npv = floating_leg - fixed_leg;
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

    std::cout << "=== Vanilla IRS AADC Pricer ===\n";
    std::cout << "Trades: " << num_trades << ", Threads: " << num_threads << "\n\n";

    // Record kernel
    auto rec_start = std::chrono::high_resolution_clock::now();
    AADCFunctions<mmType> funcs;
    IRSKernelArgs args;
    recordIRSKernel(funcs, args);
    auto rec_end = std::chrono::high_resolution_clock::now();
    double recording_time = std::chrono::duration<double>(rec_end - rec_start).count();
    double kernel_memory_mb = static_cast<double>(funcs.getMemUse() + funcs.getWorkSpaceMemUse()) / (1024.0 * 1024.0);

    std::cout << "Kernel recorded in " << std::fixed << std::setprecision(3)
              << recording_time * 1000 << " ms, memory: " << kernel_memory_mb << " MB\n\n";

    // Generate trades (all 5y semi-annual, varying notional and rate)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> notional_dist(1e6, 50e6);
    std::uniform_real_distribution<double> rate_dist(0.02, 0.06);

    struct TradeParams { double notional; double fixed_rate; };
    std::vector<TradeParams> trades(num_trades);
    for (int i = 0; i < num_trades; ++i) {
        trades[i] = {notional_dist(rng), rate_dist(rng)};
    }

    // Market data: USD curve
    std::array<double, NUM_IR_TENORS> usd_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        usd_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
    }

    // Evaluate kernel across trades
    auto eval_start = std::chrono::high_resolution_clock::now();

    std::vector<double> prices(num_trades);
    std::array<double, NUM_IR_TENORS> ir_delta_sum{};

    auto ws = std::shared_ptr<AADCWorkSpace<mmType>>(funcs.createWorkSpace());

    int num_batches = (num_trades + AVX_WIDTH - 1) / AVX_WIDTH;

    for (int batch = 0; batch < num_batches; ++batch) {
        int batch_start = batch * AVX_WIDTH;

        // Set curve rates (same for all trades in this simple example)
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            ws->setVal(args.curve_args[k], mmSetConst<mmType>(usd_rates[k]));
        }

        // Set trade parameters per lane
        mmType mm_notional, mm_rate;
        double* not_ptr = reinterpret_cast<double*>(&mm_notional);
        double* rate_ptr = reinterpret_cast<double*>(&mm_rate);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) {
                not_ptr[lane] = trades[idx].notional;
                rate_ptr[lane] = trades[idx].fixed_rate;
            } else {
                not_ptr[lane] = 0.0;
                rate_ptr[lane] = 0.0;
            }
        }
        ws->setVal(args.notional_arg, mm_notional);
        ws->setVal(args.fixed_rate_arg, mm_rate);

        // Forward pass
        funcs.forward(*ws);

        // Extract prices
        mmType mm_npv = ws->val(args.npv_res);
        double* npv_ptr = reinterpret_cast<double*>(&mm_npv);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int idx = batch_start + lane;
            if (idx < num_trades) prices[idx] = npv_ptr[lane];
        }

        // Reverse pass for sensitivities
        ws->resetDiff();
        ws->setDiff(args.npv_res, 1.0);
        funcs.reverse(*ws);

        // Accumulate IR delta sensitivities
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            mmType mm_diff = ws->diff(args.curve_args[k]);
            ir_delta_sum[k] += mmSum(mm_diff);
        }
    }

    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();

    // Results
    double portfolio_npv = 0.0;
    for (double p : prices) portfolio_npv += p;

    std::cout << "Portfolio NPV: " << std::fixed << std::setprecision(2) << portfolio_npv << "\n\n";
    std::cout << "IR Delta (AAD, summed across trades):\n";
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        std::cout << "  " << std::setw(4) << IR_TENOR_LABELS[k] << ": "
                  << std::setw(14) << ir_delta_sum[k] << "\n";
    }

    // SIMM aggregation commented out — models compute prices and risk only
    /*
    std::array<double, NUM_IR_TENORS> ir_dv01{};
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        ir_dv01[k] = ir_delta_sum[k] * 0.0001;
    }
    double ir_delta_margin = aggregateIRDelta(ir_dv01);

    SIMMResults simm_results;
    simm_results.ir_delta_margin = ir_delta_margin;
    simm_results.computeTotal();

    std::cout << "\n=== SIMM Margin ===\n";
    std::cout << "  IR Delta:   " << std::setw(14) << ir_delta_margin << "\n";
    std::cout << "  TOTAL SIMM: " << std::setw(14) << simm_results.total_margin << "\n\n";

    SIMMExecutionLogger logger("data/execution_log.csv");
    logger.log("vanilla_irs_aadc_cpp", MODEL_VERSION, "price_with_greeks",
               num_trades, num_threads, portfolio_npv, simm_results.total_margin,
               ir_delta_margin, 0.0, 0.0, 0.0, 0.0, 0.0,
               eval_time, recording_time, kernel_memory_mb,
               "C++", true, AVX_WIDTH);
    */

    std::cout << "Eval time: " << std::setprecision(3) << eval_time * 1000 << " ms\n";
    std::cout << "Recording time: " << recording_time * 1000 << " ms\n";
    std::cout << "Total time: " << (eval_time + recording_time) * 1000 << " ms\n";

    return 0;
}
