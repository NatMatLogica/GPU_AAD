// SIMM Optimizer - Full C++ AADC Implementation
// Combines: trade generation, CRIF sensitivity computation, SIMM aggregation
// kernel recording, batched multi-portfolio evaluation, gradient-based optimization.
//
// Usage:
//   ./simm_optimizer --trades 1000 --portfolios 5 --threads 8 --max-iters 100
//                    --method adam --seed 42 --verbose
//
// Version: 1.0.0
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
#include <fstream>
#include <variant>
#include <numeric>
#include <sstream>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <aadc/aadc.h>

#include "simm_params_v26.h"
#include "market_data.h"
#include "vanilla_irs.h"
#include "equity_option.h"
#include "inflation_swap.h"
#include "fx_option.h"
#include "xccy_swap.h"
#include "factor_metadata.h"
#include "sensitivity_matrix.h"
#include "simm_aggregation.h"
#include "allocation_optimizer.h"
#include "execution_logger.h"

using namespace simm;
using Trade = std::variant<IRSwapTrade, EquityOptionTrade, InflationSwapTrade,
                           FXOptionTrade, XCCYSwapTrade>;

static const char* MODEL_VERSION = "1.0.0";

// ============================================================================
// Market Environment
// ============================================================================
struct MarketEnv {
    YieldCurve<double> usd_curve;
    YieldCurve<double> eur_curve;
    InflationCurve<double> inflation;
    double equity_spot = 100.0;
    double equity_vol = 0.25;
    double fx_spot = 1.10;      // EURUSD
    double fx_vol = 0.12;
};

MarketEnv buildMarket() {
    MarketEnv mkt;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        mkt.usd_curve.zero_rates[i] = 0.04 + 0.01 * IR_TENORS[i] / 30.0;
        mkt.eur_curve.zero_rates[i] = 0.03 + 0.005 * IR_TENORS[i] / 30.0;
        mkt.inflation.inflation_rates[i] = 0.025 + 0.002 * IR_TENORS[i] / 30.0;
    }
    mkt.inflation.base_cpi = 100.0;
    return mkt;
}

// ============================================================================
// Portfolio Generation
// ============================================================================
std::vector<Trade> generatePortfolio(int num_trades, std::mt19937& rng) {
    std::vector<Trade> portfolio;
    portfolio.reserve(num_trades);

    std::uniform_int_distribution<int> type_dist(0, 4);
    std::uniform_real_distribution<double> mat_dist(1.0, 10.0);
    std::uniform_real_distribution<double> not_dist(1e6, 50e6);
    std::uniform_int_distribution<int> bool_dist(0, 1);

    for (int i = 0; i < num_trades; ++i) {
        int type = type_dist(rng);
        double notional = not_dist(rng);
        double maturity = mat_dist(rng);

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
                t.equity_bucket = 1 + (rng() % 12);
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

// ============================================================================
// Price a trade
// ============================================================================
double priceTrade(const Trade& trade, const MarketEnv& mkt) {
    return std::visit([&](const auto& t) -> double {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, IRSwapTrade>)
            return priceVanillaIRS(t, mkt.usd_curve);
        else if constexpr (std::is_same_v<T, EquityOptionTrade>)
            return priceEquityOption(t, mkt.usd_curve, mkt.equity_spot, mkt.equity_vol);
        else if constexpr (std::is_same_v<T, InflationSwapTrade>)
            return priceInflationSwap(t, mkt.usd_curve, mkt.inflation);
        else if constexpr (std::is_same_v<T, FXOptionTrade>)
            return priceFXOption(t, mkt.fx_spot, mkt.fx_vol, mkt.usd_curve, mkt.eur_curve);
        else if constexpr (std::is_same_v<T, XCCYSwapTrade>)
            return priceXCCYSwap(t, mkt.usd_curve, mkt.eur_curve, mkt.fx_spot);
        return 0.0;
    }, trade);
}

// ============================================================================
// Compute CRIF sensitivities for a single trade via bump-and-revalue
// ============================================================================
std::vector<CRIFRecord> computeTradeCRIF(const Trade& trade, const MarketEnv& mkt) {
    std::vector<CRIFRecord> crif;
    double base_price = priceTrade(trade, mkt);
    double bp = 0.0001;
    double spot_bump = 0.01;
    double vol_bump = 0.01;

    // IR Delta (USD curve) - all 12 tenors
    for (int k = 0; k < NUM_IR_TENORS; ++k) {
        MarketEnv mkt_up = mkt;
        mkt_up.usd_curve = mkt.usd_curve.bumped(k, bp);
        double sens = (priceTrade(trade, mkt_up) - base_price) / bp;
        if (std::abs(sens) > 1e-6) {
            crif.push_back({"Risk_IRCurve", "USD", 0,
                           v26::TENOR_LABELS[k], sens * bp}); // DV01
        }
    }

    // IR Delta (EUR curve) - for XCCY and FX trades
    bool has_eur = std::holds_alternative<FXOptionTrade>(trade) ||
                   std::holds_alternative<XCCYSwapTrade>(trade);
    if (has_eur) {
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            MarketEnv mkt_up = mkt;
            mkt_up.eur_curve = mkt.eur_curve.bumped(k, bp);
            double sens = (priceTrade(trade, mkt_up) - base_price) / bp;
            if (std::abs(sens) > 1e-6) {
                crif.push_back({"Risk_IRCurve", "EUR", 0,
                               v26::TENOR_LABELS[k], sens * bp});
            }
        }
    }

    // Inflation Delta
    if (std::holds_alternative<InflationSwapTrade>(trade)) {
        for (int k = 0; k < NUM_IR_TENORS; ++k) {
            MarketEnv mkt_up = mkt;
            mkt_up.inflation = mkt.inflation.bumped(k, bp);
            double sens = (priceTrade(trade, mkt_up) - base_price) / bp;
            if (std::abs(sens) > 1e-6) {
                crif.push_back({"Risk_Inflation", "USD", 0,
                               v26::TENOR_LABELS[k], sens * bp});
            }
        }
    }

    // Equity Delta
    if (std::holds_alternative<EquityOptionTrade>(trade)) {
        MarketEnv mkt_up = mkt;
        mkt_up.equity_spot *= (1.0 + spot_bump);
        double sens = (priceTrade(trade, mkt_up) - base_price) / (mkt.equity_spot * spot_bump);
        int bucket = std::get<EquityOptionTrade>(trade).equity_bucket;
        if (std::abs(sens) > 1e-6) {
            crif.push_back({"Risk_Equity", "EQ_SPOT", bucket, "", sens * mkt.equity_spot * spot_bump});
        }
    }

    // Equity Vega
    if (std::holds_alternative<EquityOptionTrade>(trade)) {
        MarketEnv mkt_up = mkt;
        mkt_up.equity_vol += vol_bump;
        double sens = (priceTrade(trade, mkt_up) - base_price) / vol_bump;
        int bucket = std::get<EquityOptionTrade>(trade).equity_bucket;
        if (std::abs(sens) > 1e-6) {
            crif.push_back({"Risk_EquityVol", "EQ_VOL", bucket, "1y", sens * vol_bump});
        }
    }

    // FX Delta
    if (std::holds_alternative<FXOptionTrade>(trade) ||
        std::holds_alternative<XCCYSwapTrade>(trade)) {
        MarketEnv mkt_up = mkt;
        mkt_up.fx_spot *= (1.0 + spot_bump);
        double sens = (priceTrade(trade, mkt_up) - base_price) / (mkt.fx_spot * spot_bump);
        if (std::abs(sens) > 1e-6) {
            crif.push_back({"Risk_FX", "EUR", 0, "", sens * mkt.fx_spot * spot_bump});
        }
    }

    // FX Vega
    if (std::holds_alternative<FXOptionTrade>(trade)) {
        MarketEnv mkt_up = mkt;
        mkt_up.fx_vol += vol_bump;
        double sens = (priceTrade(trade, mkt_up) - base_price) / vol_bump;
        if (std::abs(sens) > 1e-6) {
            crif.push_back({"Risk_FXVol", "EURUSD", 0, "1y", sens * vol_bump});
        }
    }

    return crif;
}

// ============================================================================
// Log result to CSV
// ============================================================================
// Log result matching Python's execution_log_portfolio.csv schema:
// timestamp,model_name,model_version,trade_types,num_trades,num_simm_buckets,
// num_portfolios,num_threads,crif_time_sec,crif_kernel_recording_sec,simm_time_sec,
// im_sens_time_sec,im_kernel_recording_sec,group_id,num_group_trades,im_result,
// num_im_sensitivities,reallocate_n,reallocate_time_sec,im_before_realloc,
// im_after_realloc,realloc_trades_moved,realloc_im_reduction,realloc_im_reduction_pct,
// im_realloc_estimate,realloc_estimate_matches,optimize_method,optimize_time_sec,
// optimize_initial_im,optimize_final_im,optimize_trades_moved,optimize_iterations,
// optimize_im_reduction_pct,optimize_converged,optimize_max_iters,status
void logResult(const std::string& log_path, const OptimizationResult& opt,
               int num_trades, int num_portfolios, int K, int num_threads,
               const std::string& method, const std::string& trade_types,
               double crif_time_sec, double im_kernel_recording_sec,
               double simm_time_sec, double im_sens_time_sec,
               int num_im_sensitivities, int max_iters, bool converged) {
    std::ofstream f(log_path, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "WARNING: Could not open log file: " << log_path << "\n";
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));

    double reduction_pct = 100.0 * (1.0 - opt.final_im / opt.initial_im);

    // Write one "ALL" summary row matching Python schema (36 columns)
    f << std::fixed;
    f << buf << ","                                                 // timestamp
      << "simm_optimizer_cpp," << MODEL_VERSION << ","              // model_name, model_version
      << trade_types << ","                                         // trade_types
      << num_trades << ","                                          // num_trades
      << K << ","                                                   // num_simm_buckets (= K risk factors)
      << num_portfolios << ","                                      // num_portfolios
      << num_threads << ","                                         // num_threads
      << std::setprecision(6) << crif_time_sec << ","               // crif_time_sec
      << ","                                                        // crif_kernel_recording_sec (N/A, bump&revalue)
      << std::setprecision(6) << simm_time_sec << ","               // simm_time_sec
      << std::setprecision(6) << im_sens_time_sec << ","            // im_sens_time_sec
      << std::setprecision(6) << im_kernel_recording_sec << ","     // im_kernel_recording_sec
      << "ALL,"                                                     // group_id
      << num_trades << ","                                          // num_group_trades
      << std::setprecision(2) << opt.initial_im << ","              // im_result (initial total IM)
      << num_im_sensitivities << ","                                // num_im_sensitivities
      << ",,,,,,,,,";                                               // reallocate_* (8 empty fields)
    f << method << ","                                              // optimize_method
      << std::setprecision(6) << opt.total_eval_time_sec << ","     // optimize_time_sec
      << std::setprecision(2) << opt.initial_im << ","              // optimize_initial_im
      << opt.final_im << ","                                        // optimize_final_im
      << opt.trades_moved << ","                                    // optimize_trades_moved
      << opt.num_iterations << ","                                  // optimize_iterations
      << std::setprecision(6) << reduction_pct << ","               // optimize_im_reduction_pct
      << (converged ? "True" : "False") << ","                      // optimize_converged
      << max_iters << ","                                           // optimize_max_iters
      << "success\n";                                               // status
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    // Default parameters
    int num_trades = 100;
    int num_portfolios = 5;
    int num_threads = 4;
    int max_iters = 100;
    int seed = 42;
    std::string method = "adam";
    bool verbose = true;
    bool greedy_refinement = true;
    std::string log_path = std::string(PROJECT_SOURCE_DIR) + "/data/execution_log_portfolio.csv";

    // Parse CLI
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trades" && i+1 < argc) num_trades = std::stoi(argv[++i]);
        else if (arg == "--portfolios" && i+1 < argc) num_portfolios = std::stoi(argv[++i]);
        else if (arg == "--threads" && i+1 < argc) num_threads = std::stoi(argv[++i]);
        else if (arg == "--max-iters" && i+1 < argc) max_iters = std::stoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc) seed = std::stoi(argv[++i]);
        else if (arg == "--method" && i+1 < argc) method = argv[++i];
        else if (arg == "--no-greedy") greedy_refinement = false;
        else if (arg == "--quiet") verbose = false;
        else if (arg == "--log" && i+1 < argc) log_path = argv[++i];
        else if (arg == "--help") {
            std::cout << "ISDA-SIMM Optimizer (C++ AADC)\n\n"
                      << "Usage: simm_optimizer [options]\n\n"
                      << "Options:\n"
                      << "  --trades N        Number of trades (default: 100)\n"
                      << "  --portfolios N    Number of portfolios (default: 5)\n"
                      << "  --threads N       Number of threads (default: 4)\n"
                      << "  --max-iters N     Max optimizer iterations (default: 100)\n"
                      << "  --seed N          Random seed (default: 42)\n"
                      << "  --method M        Optimizer: gradient_descent, adam (default: adam)\n"
                      << "  --no-greedy       Skip greedy refinement after continuous optimization\n"
                      << "  --quiet           Suppress verbose output\n"
                      << "  --log PATH        Log file path (default: data/execution_log_portfolio.csv)\n";
            return 0;
        }
    }

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif

    auto total_start = std::chrono::high_resolution_clock::now();

    std::cout << "================================================================================\n";
    std::cout << "              ISDA-SIMM Optimizer (C++ AADC, Full Multithreading)\n";
    std::cout << "================================================================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Trades:       " << num_trades << "\n";
    std::cout << "  Portfolios:   " << num_portfolios << "\n";
    std::cout << "  Threads:      " << num_threads << "\n";
    std::cout << "  Max iters:    " << max_iters << "\n";
    std::cout << "  Method:       " << method << "\n";
    std::cout << "  Seed:         " << seed << "\n";
    std::cout << "  Greedy:       " << (greedy_refinement ? "yes" : "no") << "\n";
    std::cout << "  SIMM version: 2.6\n\n";

    // ========== Step 1: Generate portfolio and market ==========
    std::cout << "--- Step 1: Generate portfolio ---\n";
    std::mt19937 rng(seed);
    MarketEnv mkt = buildMarket();
    std::vector<Trade> portfolio = generatePortfolio(num_trades, rng);

    int n_irs=0, n_eq=0, n_inf=0, n_fx=0, n_xccy=0;
    for (auto& t : portfolio) {
        if (std::holds_alternative<IRSwapTrade>(t)) ++n_irs;
        else if (std::holds_alternative<EquityOptionTrade>(t)) ++n_eq;
        else if (std::holds_alternative<InflationSwapTrade>(t)) ++n_inf;
        else if (std::holds_alternative<FXOptionTrade>(t)) ++n_fx;
        else if (std::holds_alternative<XCCYSwapTrade>(t)) ++n_xccy;
    }
    std::cout << "  IRS: " << n_irs << ", EQ Options: " << n_eq
              << ", Inflation: " << n_inf << ", FX Options: " << n_fx
              << ", XCCY: " << n_xccy << "\n\n";

    // ========== Step 2: Compute per-trade CRIF sensitivities ==========
    std::cout << "--- Step 2: Compute CRIF sensitivities (bump & revalue) ---\n";
    auto crif_start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> trade_ids(num_trades);
    std::vector<std::vector<CRIFRecord>> trade_crifs(num_trades);

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < num_trades; ++t) {
        trade_ids[t] = "T" + std::to_string(t);
        trade_crifs[t] = computeTradeCRIF(portfolio[t], mkt);
    }

    auto crif_end = std::chrono::high_resolution_clock::now();
    double crif_time = std::chrono::duration<double>(crif_end - crif_start).count();

    int total_crif_rows = 0;
    for (auto& c : trade_crifs) total_crif_rows += static_cast<int>(c.size());
    std::cout << "  CRIF rows: " << total_crif_rows << " (avg "
              << std::setprecision(1) << double(total_crif_rows)/num_trades << "/trade)\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << crif_time * 1000 << " ms\n\n";

    // ========== Step 3: Build sensitivity matrix ==========
    std::cout << "--- Step 3: Build sensitivity matrix ---\n";
    SensitivityMatrix S = buildSensitivityMatrix(trade_ids, trade_crifs);
    std::cout << "  Dimensions: T=" << S.T << " x K=" << S.K << "\n";

    // Compute bucket sums for concentration factors
    auto bucket_sums = computeBucketSumSensitivities(S, S.risk_factors);
    auto metadata = buildFactorMetadata(S.risk_factors, bucket_sums);

    std::cout << "  Risk factors by class: ";
    std::array<int, NUM_RISK_CLASSES> rc_counts{};
    for (auto& m : metadata) rc_counts[static_cast<int>(m.risk_class)]++;
    for (int i = 0; i < NUM_RISK_CLASSES; ++i) {
        if (rc_counts[i] > 0)
            std::cout << riskClassName(static_cast<RiskClass>(i)) << "=" << rc_counts[i] << " ";
    }
    std::cout << "\n\n";

    // ========== Step 4: Record SIMM aggregation kernel ==========
    std::cout << "--- Step 4: Record AADC SIMM kernel ---\n";
    SIMMKernel kernel;
    recordSIMMKernel(kernel, S.K, metadata);
    std::cout << "  K inputs: " << kernel.K << "\n";
    std::cout << "  Recording time: " << std::fixed << std::setprecision(2)
              << kernel.recording_time_sec * 1000 << " ms\n\n";

    // ========== Step 5: Create initial allocation ==========
    std::cout << "--- Step 5: Initialize allocation ---\n";
    AllocationMatrix alloc(num_trades, num_portfolios);
    for (int t = 0; t < num_trades; ++t) {
        alloc(t, t % num_portfolios) = 1.0;  // Round-robin
    }

    // Verify: compute initial IM
    auto init_eval = evaluateAllPortfoliosMT(kernel, S, alloc, num_threads);
    double init_total_im = std::accumulate(init_eval.ims.begin(), init_eval.ims.end(), 0.0);
    std::cout << "  Initial total IM: $" << std::fixed << std::setprecision(2) << init_total_im << "\n";
    std::cout << "  Per-portfolio IMs: ";
    for (int p = 0; p < num_portfolios; ++p) {
        std::cout << "$" << std::setprecision(2) << init_eval.ims[p];
        if (p < num_portfolios - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    // ========== Step 6: Optimize ==========
    std::cout << "--- Step 6: Optimize allocation ---\n";
    OptimizationResult opt_result;

    if (method == "gradient_descent") {
        opt_result = optimizeGradientDescent(kernel, S, alloc, num_threads,
                                              max_iters, 0.0, 1e-6, verbose);
    } else {
        opt_result = optimizeAdam(kernel, S, alloc, num_threads,
                                   max_iters, 0.0, 1e-6, verbose);
    }
    std::cout << "\n";

    // ========== Step 7: Greedy refinement on integer allocation ==========
    if (greedy_refinement) {
        std::cout << "--- Step 7: Greedy refinement ---\n";
        AllocationMatrix int_alloc = roundToInteger(opt_result.final_allocation);

        auto greedy_result = greedyLocalSearch(kernel, S, int_alloc, num_threads,
                                               50, verbose);
        // Use greedy result if better
        if (greedy_result.final_im < opt_result.final_im) {
            opt_result.final_allocation = std::move(greedy_result.final_allocation);
            opt_result.final_im = greedy_result.final_im;
            opt_result.trades_moved = countTradesMoved(alloc, opt_result.final_allocation);
            opt_result.total_eval_time_sec += greedy_result.total_eval_time_sec;
        }
        std::cout << "\n";
    }

    // ========== Results ==========
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "================================================================================\n";
    std::cout << "                              RESULTS\n";
    std::cout << "================================================================================\n";
    std::cout << std::fixed;
    std::cout << "  Initial IM:     $" << std::setprecision(2) << opt_result.initial_im << "\n";
    std::cout << "  Final IM:       $" << std::setprecision(2) << opt_result.final_im << "\n";
    std::cout << "  Reduction:      " << std::setprecision(1)
              << 100.0 * (1.0 - opt_result.final_im / opt_result.initial_im) << "%\n";
    std::cout << "  Trades moved:   " << opt_result.trades_moved << " / " << num_trades << "\n";
    std::cout << "  Iterations:     " << opt_result.num_iterations << "\n";
    std::cout << "\n  Timing:\n";
    std::cout << "    CRIF generation:   " << std::setprecision(2) << crif_time * 1000 << " ms\n";
    std::cout << "    Kernel recording:  " << opt_result.kernel_recording_time_sec * 1000 << " ms\n";
    std::cout << "    Optimization eval: " << opt_result.total_eval_time_sec * 1000 << " ms\n";
    std::cout << "    Total wall time:   " << total_time * 1000 << " ms\n";
    std::cout << "================================================================================\n";

    // ========== Log ==========
    // Build trade_types string matching Python format
    std::string trade_types;
    if (n_irs > 0) trade_types += "ir_swap";
    if (n_eq > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "equity_option"; }
    if (n_inf > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "inflation_swap"; }
    if (n_fx > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "fx_option"; }
    if (n_xccy > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "xccy_swap"; }
    // Quote if contains commas (CSV convention)
    if (trade_types.find(',') != std::string::npos)
        trade_types = "\"" + trade_types + "\"";

    // Determine convergence: check if optimizer stopped before max_iters
    bool converged = opt_result.num_iterations < max_iters;

    logResult(log_path, opt_result, num_trades, num_portfolios, S.K, num_threads,
              method, trade_types, crif_time, kernel.recording_time_sec,
              init_eval.eval_time_sec, opt_result.total_eval_time_sec,
              total_crif_rows, max_iters, converged);
    std::cout << "Logged to " << log_path << "\n";

    return 0;
}
