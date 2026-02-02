// SIMM Optimizer - Full C++ AADC Implementation
// Combines: trade generation, CRIF sensitivity computation, SIMM aggregation
// kernel recording, batched multi-portfolio evaluation, gradient-based optimization.
// Modes: optimize, attribution, whatif, pretrade
//
// Usage:
//   ./simm_optimizer --trades 1000 --portfolios 5 --threads 8 --max-iters 100
//                    --method adam --seed 42 --verbose --mode optimize --crif-method aadc
//
// Version: 2.0.0
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
#include <algorithm>
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
#include "crif_aadc.h"
#include "simm_analytics.h"
#include "data_import.h"

using namespace simm;
// Trade type alias is defined in crif_aadc.h (namespace simm)

static const char* MODEL_VERSION = "2.0.0";

// ============================================================================
// Build Market (MarketEnv defined in crif_aadc.h)
// ============================================================================
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
    std::string mode = "optimize";
    std::string crif_method = "aadc";
    int unwind_n = 5;
    double stress_factor = 1.5;
    bool verbose = true;
    bool greedy_refinement = true;
    bool pre_weighted = false;
    std::string log_path = std::string(PROJECT_SOURCE_DIR) + "/data/execution_log_portfolio.csv";
    std::string input_dir = "";

    // Parse CLI
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trades" && i+1 < argc) num_trades = std::stoi(argv[++i]);
        else if (arg == "--portfolios" && i+1 < argc) num_portfolios = std::stoi(argv[++i]);
        else if (arg == "--threads" && i+1 < argc) num_threads = std::stoi(argv[++i]);
        else if (arg == "--max-iters" && i+1 < argc) max_iters = std::stoi(argv[++i]);
        else if (arg == "--seed" && i+1 < argc) seed = std::stoi(argv[++i]);
        else if (arg == "--method" && i+1 < argc) method = argv[++i];
        else if (arg == "--mode" && i+1 < argc) mode = argv[++i];
        else if (arg == "--crif-method" && i+1 < argc) crif_method = argv[++i];
        else if (arg == "--unwind-n" && i+1 < argc) unwind_n = std::stoi(argv[++i]);
        else if (arg == "--stress-factor" && i+1 < argc) stress_factor = std::stod(argv[++i]);
        else if (arg == "--no-greedy") greedy_refinement = false;
        else if (arg == "--pre-weighted") pre_weighted = true;
        else if (arg == "--quiet") verbose = false;
        else if (arg == "--log" && i+1 < argc) log_path = argv[++i];
        else if (arg == "--input-dir" && i+1 < argc) input_dir = argv[++i];
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
                      << "  --mode M          Mode: optimize, attribution, whatif, pretrade, throughput (default: optimize)\n"
                      << "  --crif-method M   CRIF: aadc, bump (default: aadc)\n"
                      << "  --unwind-n N      Trades to unwind in whatif mode (default: 5)\n"
                      << "  --stress-factor F Shock multiplier in whatif mode (default: 1.5)\n"
                      << "  --no-greedy       Skip greedy refinement after continuous optimization\n"
                      << "  --pre-weighted    Use pre-weighted SIMM kernel (smaller tape, faster eval)\n"
                      << "  --quiet           Suppress verbose output\n"
                      << "  --log PATH        Log file path (default: data/execution_log_portfolio.csv)\n"
                      << "  --input-dir DIR   Load S matrix, metadata, allocation from DIR (Python export)\n";
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
    std::cout << "  Mode:         " << mode << "\n";
    std::cout << "  CRIF method:  " << crif_method << "\n";
    std::cout << "  Max iters:    " << max_iters << "\n";
    std::cout << "  Method:       " << method << "\n";
    std::cout << "  Seed:         " << seed << "\n";
    std::cout << "  Greedy:       " << (greedy_refinement ? "yes" : "no") << "\n";
    if (!input_dir.empty())
        std::cout << "  Input dir:    " << input_dir << "\n";
    std::cout << "  SIMM version: 2.6\n\n";

    // ========== Steps 1-3: Build or import sensitivity matrix + metadata ==========
    SensitivityMatrix S;
    std::vector<FactorMeta> metadata;
    std::vector<Trade> portfolio;  // Only populated in generate mode
    std::string trade_types;
    double crif_time = 0.0;
    int total_crif_rows = 0;

    if (!input_dir.empty()) {
        // ---- Import mode: load pre-computed data from Python ----
        std::cout << "--- Import Mode: Loading shared data from " << input_dir << " ---\n";
        auto import_start = std::chrono::high_resolution_clock::now();

        S = loadSensitivityMatrix(input_dir);
        metadata = loadFactorMetadata(input_dir);
        num_trades = S.T;

        auto import_end = std::chrono::high_resolution_clock::now();
        double import_time = std::chrono::duration<double>(import_end - import_start).count();

        std::cout << "  Dimensions: T=" << S.T << " x K=" << S.K << "\n";
        std::cout << "  Import time: " << std::fixed << std::setprecision(2)
                  << import_time * 1000 << " ms\n";

        std::cout << "  Risk factors by class: ";
        std::array<int, NUM_RISK_CLASSES> rc_counts{};
        for (auto& m : metadata) rc_counts[static_cast<int>(m.risk_class)]++;
        for (int i = 0; i < NUM_RISK_CLASSES; ++i) {
            if (rc_counts[i] > 0)
                std::cout << riskClassName(static_cast<RiskClass>(i)) << "=" << rc_counts[i] << " ";
        }
        std::cout << "\n";

        trade_types = "imported";
        std::cout << "\n";
    } else {
        // ---- Generate mode: original Steps 1-3 ----
        std::cout << "--- Step 1: Generate portfolio ---\n";
        std::mt19937 rng(seed);
        MarketEnv mkt = buildMarket();
        portfolio = generatePortfolio(num_trades, rng);

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

        // Build trade_types string
        if (n_irs > 0) trade_types += "ir_swap";
        if (n_eq > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "equity_option"; }
        if (n_inf > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "inflation_swap"; }
        if (n_fx > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "fx_option"; }
        if (n_xccy > 0) { if (!trade_types.empty()) trade_types += ","; trade_types += "xccy_swap"; }

        std::vector<std::string> trade_ids(num_trades);
        std::vector<std::vector<CRIFRecord>> trade_crifs(num_trades);
        double crif_recording_time = 0.0;

        for (int t = 0; t < num_trades; ++t)
            trade_ids[t] = "T" + std::to_string(t);

        if (crif_method == "aadc") {
            std::cout << "--- Step 2: Compute CRIF sensitivities (AADC adjoint) ---\n";
            double aadc_eval_time = 0.0;
            computeAllCRIFsAADC(portfolio, mkt, num_threads, trade_crifs,
                                crif_recording_time, aadc_eval_time);
            crif_time = aadc_eval_time;
            std::cout << "  AADC recording: " << std::fixed << std::setprecision(2)
                      << crif_recording_time * 1000 << " ms\n";
        } else {
            std::cout << "--- Step 2: Compute CRIF sensitivities (bump & revalue) ---\n";
            auto crif_start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for schedule(dynamic)
            for (int t = 0; t < num_trades; ++t) {
                trade_crifs[t] = computeTradeCRIF(portfolio[t], mkt);
            }

            auto crif_end = std::chrono::high_resolution_clock::now();
            crif_time = std::chrono::duration<double>(crif_end - crif_start).count();
        }

        total_crif_rows = 0;
        for (auto& c : trade_crifs) total_crif_rows += static_cast<int>(c.size());
        std::cout << "  CRIF rows: " << total_crif_rows << " (avg "
                  << std::setprecision(1) << double(total_crif_rows)/num_trades << "/trade)\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(2) << crif_time * 1000 << " ms\n\n";

        std::cout << "--- Step 3: Build sensitivity matrix ---\n";
        S = buildSensitivityMatrix(trade_ids, trade_crifs);
        std::cout << "  Dimensions: T=" << S.T << " x K=" << S.K << "\n";

        auto bucket_sums = computeBucketSumSensitivities(S, S.risk_factors);
        metadata = buildFactorMetadata(S.risk_factors, bucket_sums);

        std::cout << "  Risk factors by class: ";
        std::array<int, NUM_RISK_CLASSES> rc_counts{};
        for (auto& m : metadata) rc_counts[static_cast<int>(m.risk_class)]++;
        for (int i = 0; i < NUM_RISK_CLASSES; ++i) {
            if (rc_counts[i] > 0)
                std::cout << riskClassName(static_cast<RiskClass>(i)) << "=" << rc_counts[i] << " ";
        }
        std::cout << "\n\n";
    }

    // ========== Step 4: Record AADC SIMM kernel (with cache) ==========
    std::cout << "--- Step 4: Record AADC SIMM kernel ---\n";
    static SIMMKernelCache simm_cache;
    SIMMKernel& kernel = getOrRecordSIMMKernel(simm_cache, S.K, metadata, pre_weighted);
    std::cout << "  K inputs: " << kernel.K
              << (pre_weighted ? " (pre-weighted)" : "") << "\n";
    std::cout << "  Recording time: " << std::fixed << std::setprecision(2)
              << kernel.recording_time_sec * 1000 << " ms"
              << " (cache: " << simm_cache.hits() << " hits, " << simm_cache.misses() << " misses)\n\n";

    // ========== Mode Dispatch ==========

    if (mode == "attribution") {
        // ---- Attribution Mode: Euler margin decomposition ----
        std::cout << "--- Attribution Mode ---\n";
        auto attrib = computeMarginAttribution(kernel, S, num_threads);

        std::cout << std::fixed;
        std::cout << "\n  Total IM:            $" << std::setprecision(2) << attrib.total_im << "\n";
        std::cout << "  Sum contributions:   $" << std::setprecision(2) << attrib.sum_contributions << "\n";
        std::cout << "  Euler check (ratio): " << std::setprecision(6)
                  << (attrib.total_im > 0 ? attrib.sum_contributions / attrib.total_im : 0.0) << "\n";
        std::cout << "  Eval time:           " << std::setprecision(2) << attrib.eval_time_sec * 1000 << " ms\n\n";

        int show_n = std::min(20, static_cast<int>(attrib.attributions.size()));
        std::cout << "  Top " << show_n << " contributors:\n";
        std::cout << "  " << std::setw(8) << "TradeID" << std::setw(18) << "Contribution"
                  << std::setw(10) << "% of IM" << "\n";
        std::cout << "  " << std::string(36, '-') << "\n";
        for (int i = 0; i < show_n; ++i) {
            auto& a = attrib.attributions[i];
            std::cout << "  " << std::setw(8) << a.trade_id
                      << std::setw(18) << std::setprecision(2) << a.contribution
                      << std::setw(9) << std::setprecision(1) << a.pct_of_total << "%\n";
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "\n  Total wall time: " << std::setprecision(2) << total_time * 1000 << " ms\n";
        std::cout << "================================================================================\n";

    } else if (mode == "whatif") {
        // ---- What-If Mode: Unwind + Stress scenarios ----
        std::cout << "--- What-If Mode ---\n\n";
        auto eval_start = std::chrono::high_resolution_clock::now();

        // Unwind top N
        std::cout << "  [1] Unwind top " << unwind_n << " contributors:\n";
        auto unwind = whatIfUnwindTopN(kernel, S, num_threads, unwind_n);
        std::cout << "      Base IM:     $" << std::fixed << std::setprecision(2) << unwind.base_im << "\n";
        std::cout << "      Scenario IM: $" << unwind.scenario_im << "\n";
        std::cout << "      IM change:   $" << unwind.im_change
                  << " (" << std::setprecision(1) << unwind.im_change_pct << "%)\n";
        std::cout << "      Trades: ";
        for (size_t i = 0; i < std::min(size_t(5), unwind.trades_affected.size()); ++i) {
            std::cout << unwind.trades_affected[i];
            if (i + 1 < std::min(size_t(5), unwind.trades_affected.size())) std::cout << ", ";
        }
        if (unwind.trades_affected.size() > 5) std::cout << "...";
        std::cout << "\n\n";

        // Stress: scale IR by stress_factor
        std::cout << "  [2] Stress IR by " << stress_factor << "x:\n";
        auto stress = whatIfStressScenario(kernel, S, stress_factor, num_threads,
                                           metadata, RiskClass::Rates);
        std::cout << "      Base IM:     $" << std::fixed << std::setprecision(2) << stress.base_im << "\n";
        std::cout << "      Scenario IM: $" << stress.scenario_im << "\n";
        std::cout << "      IM change:   $" << stress.im_change
                  << " (" << std::setprecision(1) << stress.im_change_pct << "%)\n\n";

        // Stress: scale Equity by stress_factor
        std::cout << "  [3] Stress Equity by " << stress_factor << "x:\n";
        auto stress_eq = whatIfStressScenario(kernel, S, stress_factor, num_threads,
                                              metadata, RiskClass::Equity);
        std::cout << "      Base IM:     $" << std::fixed << std::setprecision(2) << stress_eq.base_im << "\n";
        std::cout << "      Scenario IM: $" << stress_eq.scenario_im << "\n";
        std::cout << "      IM change:   $" << stress_eq.im_change
                  << " (" << std::setprecision(1) << stress_eq.im_change_pct << "%)\n\n";

        // Marginal IM for a synthetic new trade
        std::cout << "  [4] Marginal IM (synthetic new trade):\n";
        double grad_im = 0.0;
        auto grad_k = computePortfolioGradientK(kernel, S, grad_im);
        // Create a simple new trade sensitivity vector (small IR DV01)
        std::vector<double> new_sens(S.K, 0.0);
        for (int k = 0; k < std::min(S.K, NUM_IR_TENORS); ++k) {
            new_sens[k] = 1000.0;  // $1000 DV01 per tenor
        }
        double marginal = computeMarginalIM(grad_k, new_sens, S.K);
        std::cout << "      Marginal IM: $" << std::fixed << std::setprecision(2) << marginal << "\n";

        auto eval_end = std::chrono::high_resolution_clock::now();
        double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();
        std::cout << "\n  Eval time: " << std::setprecision(2) << eval_time * 1000 << " ms\n";

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "  Total wall time: " << std::setprecision(2) << total_time * 1000 << " ms\n";
        std::cout << "================================================================================\n";

    } else if (mode == "pretrade") {
        // ---- Pre-Trade Mode: Counterparty routing + Bilateral vs Cleared ----
        std::cout << "--- Pre-Trade Mode ---\n\n";

        // Load or create initial allocation
        AllocationMatrix alloc(0, 0);
        if (!input_dir.empty()) {
            alloc = loadAllocation(input_dir);
            num_portfolios = alloc.P;
        } else {
            alloc = AllocationMatrix(num_trades, num_portfolios);
            for (int t = 0; t < num_trades; ++t)
                alloc(t, t % num_portfolios) = 1.0;
        }

        // Generate a synthetic new trade's CRIF
        std::cout << "  Generating synthetic new trade CRIF...\n";
        MarketEnv mkt_pt = buildMarket();
        std::mt19937 rng2(seed + 999);
        auto new_trades = generatePortfolio(1, rng2);
        auto new_crif = computeTradeCRIF(new_trades[0], mkt_pt);

        // Counterparty routing
        std::cout << "  [1] Counterparty Routing (across " << num_portfolios << " portfolios):\n";
        auto routing = counterpartyRouting(kernel, S, alloc, new_crif, num_threads);
        for (int p = 0; p < num_portfolios; ++p) {
            std::cout << "      Portfolio " << p << ": marginal IM = $"
                      << std::fixed << std::setprecision(2) << routing.marginal_ims[p];
            if (p == routing.best_portfolio) std::cout << "  <-- BEST";
            std::cout << "\n";
        }
        std::cout << "      Best portfolio: " << routing.best_portfolio
                  << " (marginal IM = $" << std::setprecision(2) << routing.best_marginal_im << ")\n";
        std::cout << "      Eval time: " << std::setprecision(2) << routing.eval_time_sec * 1000 << " ms\n\n";

        // Bilateral vs Cleared (requires trade objects, skip in import mode)
        if (!input_dir.empty()) {
            std::cout << "  [2] Bilateral vs Cleared: skipped (import mode)\n";
        } else {
            std::cout << "  [2] Bilateral vs Cleared:\n";
            auto clearing = bilateralVsCleared(kernel, S, portfolio, num_threads);
            std::cout << "      SIMM bilateral: $" << std::fixed << std::setprecision(2) << clearing.simm_margin << "\n";
            std::cout << "      CCP cleared:    $" << clearing.ccp_margin << "\n";
            std::cout << "      Difference:     $" << clearing.margin_difference
                      << " (" << std::setprecision(1) << clearing.difference_pct << "%)\n";
            std::cout << "      Recommendation: " << clearing.recommendation << "\n";
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "\n  Total wall time: " << std::setprecision(2) << total_time * 1000 << " ms\n";
        std::cout << "================================================================================\n";

    } else if (mode == "optimize") {
        // ---- Optimize Mode ----

        // ========== Step 5: Create initial allocation ==========
        std::cout << "--- Step 5: Initialize allocation ---\n";
        AllocationMatrix alloc(0, 0);
        if (!input_dir.empty()) {
            alloc = loadAllocation(input_dir);
            num_portfolios = alloc.P;
        } else {
            alloc = AllocationMatrix(num_trades, num_portfolios);
            for (int t = 0; t < num_trades; ++t) {
                alloc(t, t % num_portfolios) = 1.0;  // Round-robin
            }
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
        // Quote trade_types if contains commas (CSV convention)
        std::string log_trade_types = trade_types;
        if (log_trade_types.find(',') != std::string::npos)
            log_trade_types = "\"" + log_trade_types + "\"";

        // Determine convergence: check if optimizer stopped before max_iters
        bool converged = opt_result.num_iterations < max_iters;

        logResult(log_path, opt_result, num_trades, num_portfolios, S.K, num_threads,
                  method, log_trade_types, crif_time, kernel.recording_time_sec,
                  init_eval.eval_time_sec, opt_result.total_eval_time_sec,
                  total_crif_rows, max_iters, converged);
        std::cout << "Logged to " << log_path << "\n";

    } else if (mode == "throughput") {
        // ---- Throughput Mode: Measure pure kernel evals/sec ----
        std::cout << "--- Throughput Mode ---\n";

        // Create allocation (round-robin or import)
        AllocationMatrix alloc(0, 0);
        if (!input_dir.empty()) {
            alloc = loadAllocation(input_dir);
            num_portfolios = alloc.P;
        } else {
            alloc = AllocationMatrix(num_trades, num_portfolios);
            for (int t = 0; t < num_trades; ++t)
                alloc(t, t % num_portfolios) = 1.0;
        }

        int K = kernel.K;
        int T = S.T;
        int P = num_portfolios;
        int n_warmup = 5;
        int n_timed = max_iters;  // Use --max-iters as number of timed evaluations
        std::mt19937 rng(seed);
        std::normal_distribution<double> noise(0.0, 0.001);

        // Pre-compute agg_S once for warmup
        std::vector<double> agg_S(K * P);
        matmulATB(S.data.data(), alloc.data.data(), agg_S.data(), T, K, P);

        // Create workspaces
        int num_batches = (P + AVX_WIDTH - 1) / AVX_WIDTH;
        int actual_threads = std::min(num_threads, num_batches);
        std::vector<std::shared_ptr<AADCWorkSpace<mmType>>> workspaces(actual_threads);
        for (int i = 0; i < actual_threads; ++i) {
            workspaces[i] = kernel.funcs.createWorkSpace();
        }

        std::cout << "  Warmup: " << n_warmup << " evals\n";
        std::cout << "  Timed:  " << n_timed << " evals\n";
        std::cout << "  T=" << T << " K=" << K << " P=" << P
                  << " threads=" << num_threads << "\n\n";

        // Lambda: run AADC kernel only (no matrix muls)
        auto run_kernel_only = [&](const std::vector<double>& agg) {
            std::vector<double> ims(P, 0.0);
            std::vector<double> grad_k(K * P, 0.0);

            #pragma omp parallel for num_threads(actual_threads) schedule(dynamic)
            for (int batch = 0; batch < num_batches; ++batch) {
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                auto& ws = *workspaces[tid];
                int p_start = batch * AVX_WIDTH;

                for (int k = 0; k < K; ++k) {
                    mmType mm_val;
                    double* ptr = reinterpret_cast<double*>(&mm_val);
                    for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                        int p = p_start + lane;
                        ptr[lane] = (p < P) ? agg[k * P + p] : 0.0;
                    }
                    ws.setVal(kernel.sens_handles[k], mm_val);
                }
                kernel.funcs.forward(ws);
                {
                    mmType mm_im = ws.val(kernel.im_output);
                    double* im_ptr = reinterpret_cast<double*>(&mm_im);
                    for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                        int p = p_start + lane;
                        if (p < P) ims[p] = im_ptr[lane];
                    }
                }
                ws.resetDiff();
                ws.setDiff(kernel.im_output, 1.0);
                kernel.funcs.reverse(ws);
                for (int k = 0; k < K; ++k) {
                    mmType mm_grad = ws.diff(kernel.sens_handles[k]);
                    double* grad_ptr = reinterpret_cast<double*>(&mm_grad);
                    for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                        int p = p_start + lane;
                        if (p < P) grad_k[k * P + p] = grad_ptr[lane];
                    }
                }
            }
            return std::make_pair(ims, grad_k);
        };

        // Warmup (kernel only)
        for (int i = 0; i < n_warmup; ++i) {
            run_kernel_only(agg_S);
        }

        // ----- Measure kernel-only times -----
        std::vector<double> kernel_times(n_timed);
        std::vector<double> agg_S_noisy(K * P);
        for (int i = 0; i < n_timed; ++i) {
            // Add tiny noise to aggregated inputs
            for (int j = 0; j < K * P; ++j)
                agg_S_noisy[j] = agg_S[j] * (1.0 + noise(rng));

            auto t0 = std::chrono::high_resolution_clock::now();
            run_kernel_only(agg_S_noisy);
            auto t1 = std::chrono::high_resolution_clock::now();
            kernel_times[i] = std::chrono::duration<double>(t1 - t0).count();
        }

        // ----- Measure full eval (matmul + kernel + chain rule) -----
        SensitivityMatrix S_noisy;
        S_noisy.T = S.T; S_noisy.K = S.K;
        S_noisy.risk_factors = S.risk_factors;
        S_noisy.trade_ids = S.trade_ids;
        S_noisy.data = S.data;

        // Warmup full
        for (int i = 0; i < n_warmup; ++i)
            evaluateAllPortfoliosMT(kernel, S_noisy, alloc, num_threads);

        std::vector<double> full_times(n_timed);
        for (int i = 0; i < n_timed; ++i) {
            for (size_t j = 0; j < S_noisy.data.size(); ++j)
                S_noisy.data[j] = S.data[j] * (1.0 + noise(rng));
            auto result = evaluateAllPortfoliosMT(kernel, S_noisy, alloc, num_threads);
            full_times[i] = result.eval_time_sec;
        }

        // Statistics
        auto stats = [](std::vector<double>& times) {
            std::sort(times.begin(), times.end());
            int n = times.size();
            double median = times[n / 2];
            double total = 0; for (double t : times) total += t;
            return std::make_tuple(median, total / n, times[0], times.back());
        };

        auto [k_med, k_mean, k_min, k_max] = stats(kernel_times);
        auto [f_med, f_mean, f_min, f_max] = stats(full_times);

        std::cout << std::fixed;
        std::cout << "  AADC kernel only (forward+reverse, no matmul):\n";
        std::cout << "    Median:      " << std::setprecision(3) << k_med * 1000 << " ms\n";
        std::cout << "    Mean:        " << std::setprecision(3) << k_mean * 1000 << " ms\n";
        std::cout << "    Evals/sec:   " << std::setprecision(1) << 1.0 / k_med << "\n";
        std::cout << "\n  Full eval (matmul + kernel + chain rule):\n";
        std::cout << "    Median:      " << std::setprecision(3) << f_med * 1000 << " ms\n";
        std::cout << "    Mean:        " << std::setprecision(3) << f_mean * 1000 << " ms\n";
        std::cout << "    Evals/sec:   " << std::setprecision(1) << 1.0 / f_med << "\n";
        std::cout << "    Matmul overhead: " << std::setprecision(1)
                  << (f_med - k_med) / f_med * 100 << "%\n";
        std::cout << "\n  Median eval:   " << std::setprecision(3) << k_med * 1000 << " ms\n";
        std::cout << "  Evals/sec:     " << std::setprecision(1) << 1.0 / k_med << "\n";
        std::cout << "  Recording:     " << std::setprecision(2) << kernel.recording_time_sec * 1000 << " ms\n";

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "  Total wall:    " << std::setprecision(2) << total_time * 1000 << " ms\n";
        std::cout << "================================================================================\n";

    } else if (mode == "all") {
        // ================================================================
        // ALL Mode: Run attribution + whatif + pretrade + optimize
        // in a SINGLE process (shared import, shared kernel).
        // Output uses [SECTION] markers for Python parsing.
        // ================================================================
        std::cout << "--- All Mode (Combined Workflow) ---\n\n";

        // Load or create allocation
        AllocationMatrix alloc(0, 0);
        if (!input_dir.empty()) {
            alloc = loadAllocation(input_dir);
            num_portfolios = alloc.P;
        } else {
            alloc = AllocationMatrix(num_trades, num_portfolios);
            for (int t = 0; t < num_trades; ++t)
                alloc(t, t % num_portfolios) = 1.0;
        }

        // [ATTRIBUTION]
        std::cout << "[ATTRIBUTION]\n";
        {
            auto attrib = computeMarginAttribution(kernel, S, num_threads);
            std::cout << std::fixed;
            std::cout << "  Total IM: $" << std::setprecision(2) << attrib.total_im << "\n";
            std::cout << "  Euler check (ratio): " << std::setprecision(6)
                      << (attrib.total_im > 0 ? attrib.sum_contributions / attrib.total_im : 0.0) << "\n";
            std::cout << "  Eval time: " << std::setprecision(2) << attrib.eval_time_sec * 1000 << " ms\n\n";
        }

        // [WHATIF]
        std::cout << "[WHATIF]\n";
        {
            auto wi_start = std::chrono::high_resolution_clock::now();

            auto unwind = whatIfUnwindTopN(kernel, S, num_threads, unwind_n);
            std::cout << "  Unwind top " << unwind_n << ": Base=$"
                      << std::fixed << std::setprecision(2) << unwind.base_im
                      << " Scenario=$" << unwind.scenario_im
                      << " (" << std::setprecision(1) << unwind.im_change_pct << "%)\n";

            auto stress = whatIfStressScenario(kernel, S, stress_factor, num_threads,
                                               metadata, RiskClass::Rates);
            std::cout << "  Stress IR " << stress_factor << "x: Scenario=$"
                      << std::fixed << std::setprecision(2) << stress.scenario_im
                      << " (" << std::setprecision(1) << stress.im_change_pct << "%)\n";

            auto stress_eq = whatIfStressScenario(kernel, S, stress_factor, num_threads,
                                                  metadata, RiskClass::Equity);
            std::cout << "  Stress Equity " << stress_factor << "x: Scenario=$"
                      << std::fixed << std::setprecision(2) << stress_eq.scenario_im
                      << " (" << std::setprecision(1) << stress_eq.im_change_pct << "%)\n";

            double grad_im = 0.0;
            auto grad_k = computePortfolioGradientK(kernel, S, grad_im);
            std::vector<double> new_sens(S.K, 0.0);
            for (int k = 0; k < std::min(S.K, NUM_IR_TENORS); ++k)
                new_sens[k] = 1000.0;
            double marginal = computeMarginalIM(grad_k, new_sens, S.K);
            std::cout << "  Marginal IM: $" << std::fixed << std::setprecision(2) << marginal << "\n";

            auto wi_end = std::chrono::high_resolution_clock::now();
            double wi_time = std::chrono::duration<double>(wi_end - wi_start).count();
            std::cout << "  Eval time: " << std::setprecision(2) << wi_time * 1000 << " ms\n\n";
        }

        // [PRETRADE]
        std::cout << "[PRETRADE]\n";
        {
            MarketEnv mkt_pt = buildMarket();
            std::mt19937 rng2(seed + 999);
            auto new_trades_pt = generatePortfolio(1, rng2);
            auto new_crif = computeTradeCRIF(new_trades_pt[0], mkt_pt);

            auto routing = counterpartyRouting(kernel, S, alloc, new_crif, num_threads);
            std::cout << "  Best portfolio: " << routing.best_portfolio
                      << " (marginal IM=$" << std::fixed << std::setprecision(2) << routing.best_marginal_im << ")\n";
            std::cout << "  Eval time: " << std::setprecision(2) << routing.eval_time_sec * 1000 << " ms\n\n";
        }

        // [OPTIMIZE]
        std::cout << "[OPTIMIZE]\n";
        {
            auto opt_wall_start = std::chrono::high_resolution_clock::now();

            auto init_eval = evaluateAllPortfoliosMT(kernel, S, alloc, num_threads);
            double init_total_im = std::accumulate(init_eval.ims.begin(), init_eval.ims.end(), 0.0);
            std::cout << "  Initial IM: $" << std::fixed << std::setprecision(2) << init_total_im << "\n";

            OptimizationResult opt_result;
            if (method == "gradient_descent") {
                opt_result = optimizeGradientDescent(kernel, S, alloc, num_threads,
                                                      max_iters, 0.0, 1e-6, verbose);
            } else {
                opt_result = optimizeAdam(kernel, S, alloc, num_threads,
                                           max_iters, 0.0, 1e-6, verbose);
            }

            int total_evals = opt_result.total_evals + 1;  // +1 for init_eval above
            int greedy_rounds = 0;

            if (greedy_refinement) {
                AllocationMatrix int_alloc = roundToInteger(opt_result.final_allocation);
                auto greedy_result = greedyLocalSearch(kernel, S, int_alloc, num_threads, 50, verbose);
                if (greedy_result.final_im < opt_result.final_im) {
                    opt_result.final_allocation = std::move(greedy_result.final_allocation);
                    opt_result.final_im = greedy_result.final_im;
                    opt_result.trades_moved = countTradesMoved(alloc, opt_result.final_allocation);
                    opt_result.total_eval_time_sec += greedy_result.total_eval_time_sec;
                }
                total_evals += greedy_result.total_evals;
                greedy_rounds = greedy_result.greedy_rounds;
            }

            auto opt_wall_end = std::chrono::high_resolution_clock::now();
            double opt_wall_sec = std::chrono::duration<double>(opt_wall_end - opt_wall_start).count();

            std::cout << "  Final IM: $" << std::setprecision(2) << opt_result.final_im << "\n";
            std::cout << "  Trades moved: " << opt_result.trades_moved << "\n";
            std::cout << "  Iterations: " << opt_result.num_iterations << "\n";
            std::cout << "  Greedy rounds: " << greedy_rounds << "\n";
            std::cout << "  Total evals: " << total_evals << "\n";
            std::cout << "  Optimization eval: " << std::setprecision(2)
                      << opt_result.total_eval_time_sec * 1000 << " ms\n";
            std::cout << "  Optimization wall: " << std::setprecision(2)
                      << opt_wall_sec * 1000 << " ms\n\n";
        }

        // Summary
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(total_end - total_start).count();
        std::cout << "Recording time: " << std::fixed << std::setprecision(2)
                  << kernel.recording_time_sec * 1000 << " ms\n";
        std::cout << "Total wall time: " << std::setprecision(2) << total_time * 1000 << " ms\n";
        std::cout << "================================================================================\n";
    }

    return 0;
}
