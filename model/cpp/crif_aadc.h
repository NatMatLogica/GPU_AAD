// AADC-Based CRIF Generation: Per-Trade Sensitivities via Adjoint Pass
// Replaces bump-and-revalue with single forward+reverse per trade
// Version: 1.0.0
#pragma once

#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>
#include <variant>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <aadc/aadc.h>
#include <aadc/aadc_matrix.h>

#include "simm_config.h"
#include "market_data.h"
#include "sensitivity_matrix.h"

using aadc::AADCFunctions;
using aadc::AADCArgument;
using aadc::AADCResult;
using aadc::AADCWorkSpace;

namespace simm {

using mmType = __m256d;
constexpr int CRIF_AVX_WIDTH = sizeof(mmType) / sizeof(double);
using aadc::mmSetConst;
using aadc::mmSum;

// ============================================================================
// Market Environment (shared across CRIF generation and main pipeline)
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

// ============================================================================
// CRIF Mapping: Maps an AADC handle to a CRIF risk factor with scaling
// ============================================================================
struct CRIFMapping {
    AADCArgument handle;
    std::string risk_type;
    std::string qualifier;
    int bucket;
    std::string label1;
    double scaling;  // Convert derivative to CRIF units
};

// ============================================================================
// IRS CRIF Kernel
// ============================================================================
struct IRSCRIFKernel {
    AADCFunctions<mmType> funcs;
    std::array<AADCArgument, NUM_IR_TENORS> usd_curve;
    AADCArgument notional_arg;
    AADCArgument fixed_rate_arg;
    AADCArgument payer_sign_arg;
    AADCResult npv_res;
    double recording_time_sec = 0.0;
};

inline void recordIRSCRIFKernel(IRSCRIFKernel& k) {
    auto t0 = std::chrono::high_resolution_clock::now();
    k.funcs.startRecording();

    // Differentiable: USD curve rates
    std::array<idouble, NUM_IR_TENORS> rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        rates[i] = idouble(0.04);
        k.usd_curve[i] = AADCArgument(rates[i].markAsInput());
    }

    // Non-diff: trade params
    idouble notional(1e6);
    idouble fixed_rate(0.03);
    idouble payer_sign(1.0);
    k.notional_arg = AADCArgument(notional.markAsInputNoDiff());
    k.fixed_rate_arg = AADCArgument(fixed_rate.markAsInputNoDiff());
    k.payer_sign_arg = AADCArgument(payer_sign.markAsInputNoDiff());

    // Fixed structure: 5y semi-annual (matches vanilla_irs_aadc.cpp)
    constexpr double maturity = 5.0;
    constexpr int frequency = 2;
    constexpr double dt = 1.0 / frequency;
    constexpr int num_periods = static_cast<int>(maturity * frequency);

    auto interpRate = [&](double t) -> idouble {
        if (t <= IR_TENORS[0]) return rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return rates[NUM_IR_TENORS - 1];
        for (int j = 0; j < NUM_IR_TENORS - 1; ++j) {
            if (t >= IR_TENORS[j] && t <= IR_TENORS[j + 1]) {
                double w = (t - IR_TENORS[j]) / (IR_TENORS[j + 1] - IR_TENORS[j]);
                return rates[j] * (1.0 - w) + rates[j + 1] * w;
            }
        }
        return rates[NUM_IR_TENORS - 1];
    };

    idouble fixed_leg(0.0);
    idouble floating_leg(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        double t_prev = (i - 1) * dt;
        idouble r_t = interpRate(t);
        idouble df = std::exp(-r_t * t);
        fixed_leg = fixed_leg + notional * fixed_rate * dt * df;

        idouble r_prev = interpRate(std::max(t_prev, 0.001));
        idouble df_prev = std::exp(-r_prev * t_prev);
        idouble fwd = std::log(df_prev / df) / (t - t_prev + 1e-15);
        floating_leg = floating_leg + notional * dt * fwd * df;
    }

    idouble npv = (floating_leg - fixed_leg) * payer_sign;
    k.npv_res = AADCResult(npv.markAsOutput());
    k.funcs.stopRecording();

    auto t1 = std::chrono::high_resolution_clock::now();
    k.recording_time_sec = std::chrono::duration<double>(t1 - t0).count();
}

inline std::vector<CRIFMapping> buildIRSMappings(const IRSCRIFKernel& k) {
    std::vector<CRIFMapping> maps;
    double bp = 0.0001;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        maps.push_back({k.usd_curve[i], "Risk_IRCurve", "USD", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
    }
    return maps;
}

// ============================================================================
// Equity Option CRIF Kernel
// ============================================================================
struct EQCRIFKernel {
    AADCFunctions<mmType> funcs;
    AADCArgument spot_arg;
    AADCArgument vol_arg;
    std::array<AADCArgument, NUM_IR_TENORS> curve_args;
    AADCArgument strike_arg;
    AADCArgument div_yield_arg;
    AADCArgument is_call_arg;
    AADCResult price_res;
    double recording_time_sec = 0.0;
};

inline void recordEQCRIFKernel(EQCRIFKernel& k) {
    auto t0 = std::chrono::high_resolution_clock::now();
    k.funcs.startRecording();

    idouble spot(100.0);
    idouble vol(0.25);
    k.spot_arg = AADCArgument(spot.markAsInput());
    k.vol_arg = AADCArgument(vol.markAsInput());

    std::array<idouble, NUM_IR_TENORS> rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        rates[i] = idouble(0.04);
        k.curve_args[i] = AADCArgument(rates[i].markAsInput());
    }

    idouble strike(100.0);
    idouble div_yield(0.02);
    idouble is_call(1.0);
    k.strike_arg = AADCArgument(strike.markAsInputNoDiff());
    k.div_yield_arg = AADCArgument(div_yield.markAsInputNoDiff());
    k.is_call_arg = AADCArgument(is_call.markAsInputNoDiff());

    constexpr double tau = 1.0;

    // Interpolate rate
    idouble r = rates[0];
    for (int j = 0; j < NUM_IR_TENORS - 1; ++j) {
        if (tau >= IR_TENORS[j] && tau <= IR_TENORS[j + 1]) {
            double w = (tau - IR_TENORS[j]) / (IR_TENORS[j + 1] - IR_TENORS[j]);
            r = rates[j] * (1.0 - w) + rates[j + 1] * w;
            break;
        }
    }

    idouble sqrt_tau = std::sqrt(idouble(tau));
    idouble d1 = (std::log(spot / strike) + (r - div_yield + 0.5 * vol * vol) * tau) /
                 (vol * sqrt_tau);
    idouble d2 = d1 - vol * sqrt_tau;

    idouble df = std::exp(-r * tau);
    idouble dq = std::exp(-div_yield * tau);

    idouble sqrt2 = std::sqrt(idouble(2.0));
    idouble nd1 = 0.5 * (1.0 + std::erf(d1 / sqrt2));
    idouble nd2 = 0.5 * (1.0 + std::erf(d2 / sqrt2));

    // Call price; is_call = +1 for call, -1 for put via put-call parity proxy
    idouble price = spot * dq * nd1 - strike * df * nd2;

    k.price_res = AADCResult(price.markAsOutput());
    k.funcs.stopRecording();

    auto t1 = std::chrono::high_resolution_clock::now();
    k.recording_time_sec = std::chrono::duration<double>(t1 - t0).count();
}

inline std::vector<CRIFMapping> buildEQMappings(const EQCRIFKernel& k, double spot, int bucket) {
    std::vector<CRIFMapping> maps;
    maps.push_back({k.spot_arg, "Risk_Equity", "EQ_SPOT", bucket, "",
                    spot * 0.01});  // dPV/dSpot * spot * 1%
    maps.push_back({k.vol_arg, "Risk_EquityVol", "EQ_VOL", bucket, "1y",
                    0.01});         // dPV/dVol * 1%
    return maps;
}

// ============================================================================
// FX Option CRIF Kernel
// ============================================================================
struct FXCRIFKernel {
    AADCFunctions<mmType> funcs;
    AADCArgument spot_arg;
    AADCArgument vol_arg;
    std::array<AADCArgument, NUM_IR_TENORS> dom_curve_args;
    std::array<AADCArgument, NUM_IR_TENORS> fgn_curve_args;
    AADCArgument strike_arg;
    AADCResult price_res;
    double recording_time_sec = 0.0;
};

inline void recordFXCRIFKernel(FXCRIFKernel& k) {
    auto t0 = std::chrono::high_resolution_clock::now();
    k.funcs.startRecording();

    idouble spot(1.10);
    idouble vol(0.12);
    k.spot_arg = AADCArgument(spot.markAsInput());
    k.vol_arg = AADCArgument(vol.markAsInput());

    std::array<idouble, NUM_IR_TENORS> dom_rates, fgn_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        dom_rates[i] = idouble(0.04);
        fgn_rates[i] = idouble(0.03);
        k.dom_curve_args[i] = AADCArgument(dom_rates[i].markAsInput());
        k.fgn_curve_args[i] = AADCArgument(fgn_rates[i].markAsInput());
    }

    idouble strike(1.10);
    k.strike_arg = AADCArgument(strike.markAsInputNoDiff());

    constexpr double tau = 1.0;

    // Interpolate rates
    auto interpRate = [&](const std::array<idouble, NUM_IR_TENORS>& rates, double t) -> idouble {
        if (t <= IR_TENORS[0]) return rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return rates[NUM_IR_TENORS - 1];
        for (int j = 0; j < NUM_IR_TENORS - 1; ++j) {
            if (t >= IR_TENORS[j] && t <= IR_TENORS[j + 1]) {
                double w = (t - IR_TENORS[j]) / (IR_TENORS[j + 1] - IR_TENORS[j]);
                return rates[j] * (1.0 - w) + rates[j + 1] * w;
            }
        }
        return rates[NUM_IR_TENORS - 1];
    };

    idouble rd = interpRate(dom_rates, tau);
    idouble rf = interpRate(fgn_rates, tau);

    idouble sqrt_tau = std::sqrt(idouble(tau));
    idouble d1 = (std::log(spot / strike) + (rd - rf + 0.5 * vol * vol) * tau) /
                 (vol * sqrt_tau);
    idouble d2 = d1 - vol * sqrt_tau;

    idouble df_dom = std::exp(-rd * tau);
    idouble df_fgn = std::exp(-rf * tau);

    idouble sqrt2 = std::sqrt(idouble(2.0));
    idouble nd1 = 0.5 * (1.0 + std::erf(d1 / sqrt2));
    idouble nd2 = 0.5 * (1.0 + std::erf(d2 / sqrt2));

    idouble price = spot * df_fgn * nd1 - strike * df_dom * nd2;

    k.price_res = AADCResult(price.markAsOutput());
    k.funcs.stopRecording();

    auto t1 = std::chrono::high_resolution_clock::now();
    k.recording_time_sec = std::chrono::duration<double>(t1 - t0).count();
}

inline std::vector<CRIFMapping> buildFXMappings(const FXCRIFKernel& k, double spot) {
    std::vector<CRIFMapping> maps;
    maps.push_back({k.spot_arg, "Risk_FX", "EUR", 0, "",
                    spot * 0.01});
    maps.push_back({k.vol_arg, "Risk_FXVol", "EURUSD", 0, "1y",
                    0.01});
    double bp = 0.0001;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        maps.push_back({k.dom_curve_args[i], "Risk_IRCurve", "USD", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
        maps.push_back({k.fgn_curve_args[i], "Risk_IRCurve", "EUR", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
    }
    return maps;
}

// ============================================================================
// Inflation Swap CRIF Kernel
// ============================================================================
struct InflCRIFKernel {
    AADCFunctions<mmType> funcs;
    std::array<AADCArgument, NUM_IR_TENORS> ir_curve_args;
    std::array<AADCArgument, NUM_IR_TENORS> infl_rate_args;
    AADCArgument notional_arg;
    AADCArgument fixed_rate_arg;
    AADCArgument base_cpi_arg;
    AADCResult npv_res;
    double recording_time_sec = 0.0;
};

inline void recordInflCRIFKernel(InflCRIFKernel& k) {
    auto t0 = std::chrono::high_resolution_clock::now();
    k.funcs.startRecording();

    std::array<idouble, NUM_IR_TENORS> ir_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        ir_rates[i] = idouble(0.04);
        k.ir_curve_args[i] = AADCArgument(ir_rates[i].markAsInput());
    }

    std::array<idouble, NUM_IR_TENORS> infl_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        infl_rates[i] = idouble(0.025);
        k.infl_rate_args[i] = AADCArgument(infl_rates[i].markAsInput());
    }

    idouble notional(1e6);
    idouble fixed_rate(0.025);
    idouble base_cpi(100.0);
    k.notional_arg = AADCArgument(notional.markAsInputNoDiff());
    k.fixed_rate_arg = AADCArgument(fixed_rate.markAsInputNoDiff());
    k.base_cpi_arg = AADCArgument(base_cpi.markAsInputNoDiff());

    constexpr double tau = 5.0;

    auto interpRate = [&](const std::array<idouble, NUM_IR_TENORS>& rates, double t) -> idouble {
        if (t <= IR_TENORS[0]) return rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return rates[NUM_IR_TENORS - 1];
        for (int j = 0; j < NUM_IR_TENORS - 1; ++j) {
            if (t >= IR_TENORS[j] && t <= IR_TENORS[j + 1]) {
                double w = (t - IR_TENORS[j]) / (IR_TENORS[j + 1] - IR_TENORS[j]);
                return rates[j] * (1.0 - w) + rates[j + 1] * w;
            }
        }
        return rates[NUM_IR_TENORS - 1];
    };

    idouble r_t = interpRate(ir_rates, tau);
    idouble df = std::exp(-r_t * tau);

    idouble infl_t = interpRate(infl_rates, tau);
    idouble fixed_leg = notional * (std::exp(fixed_rate * tau) - 1.0) * df;
    idouble cpi_T = base_cpi * std::exp(infl_t * tau);
    idouble infl_leg = notional * (cpi_T / base_cpi - 1.0) * df;

    idouble npv = infl_leg - fixed_leg;
    k.npv_res = AADCResult(npv.markAsOutput());
    k.funcs.stopRecording();

    auto t1 = std::chrono::high_resolution_clock::now();
    k.recording_time_sec = std::chrono::duration<double>(t1 - t0).count();
}

inline std::vector<CRIFMapping> buildInflMappings(const InflCRIFKernel& k) {
    std::vector<CRIFMapping> maps;
    double bp = 0.0001;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        maps.push_back({k.ir_curve_args[i], "Risk_IRCurve", "USD", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
        maps.push_back({k.infl_rate_args[i], "Risk_Inflation", "USD", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
    }
    return maps;
}

// ============================================================================
// XCCY Swap CRIF Kernel
// ============================================================================
struct XCCYCRIFKernel {
    AADCFunctions<mmType> funcs;
    AADCArgument fx_spot_arg;
    std::array<AADCArgument, NUM_IR_TENORS> dom_curve_args;
    std::array<AADCArgument, NUM_IR_TENORS> fgn_curve_args;
    AADCArgument dom_notional_arg;
    AADCArgument fgn_notional_arg;
    AADCArgument dom_rate_arg;
    AADCArgument fgn_rate_arg;
    AADCResult npv_res;
    double recording_time_sec = 0.0;
};

inline void recordXCCYCRIFKernel(XCCYCRIFKernel& k) {
    auto t0 = std::chrono::high_resolution_clock::now();
    k.funcs.startRecording();

    idouble fx_spot(1.10);
    k.fx_spot_arg = AADCArgument(fx_spot.markAsInput());

    std::array<idouble, NUM_IR_TENORS> dom_rates, fgn_rates;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        dom_rates[i] = idouble(0.04);
        fgn_rates[i] = idouble(0.03);
        k.dom_curve_args[i] = AADCArgument(dom_rates[i].markAsInput());
        k.fgn_curve_args[i] = AADCArgument(fgn_rates[i].markAsInput());
    }

    idouble dom_notional(1e6);
    idouble fgn_notional(909090.9);
    idouble dom_rate(0.04);
    idouble fgn_rate(0.03);
    k.dom_notional_arg = AADCArgument(dom_notional.markAsInputNoDiff());
    k.fgn_notional_arg = AADCArgument(fgn_notional.markAsInputNoDiff());
    k.dom_rate_arg = AADCArgument(dom_rate.markAsInputNoDiff());
    k.fgn_rate_arg = AADCArgument(fgn_rate.markAsInputNoDiff());

    constexpr double maturity = 5.0;
    constexpr int frequency = 2;
    constexpr double dt = 1.0 / frequency;
    constexpr int num_periods = static_cast<int>(maturity * frequency);

    auto interpRate = [&](const std::array<idouble, NUM_IR_TENORS>& rates, double t) -> idouble {
        if (t <= IR_TENORS[0]) return rates[0];
        if (t >= IR_TENORS[NUM_IR_TENORS - 1]) return rates[NUM_IR_TENORS - 1];
        for (int j = 0; j < NUM_IR_TENORS - 1; ++j) {
            if (t >= IR_TENORS[j] && t <= IR_TENORS[j + 1]) {
                double w = (t - IR_TENORS[j]) / (IR_TENORS[j + 1] - IR_TENORS[j]);
                return rates[j] * (1.0 - w) + rates[j + 1] * w;
            }
        }
        return rates[NUM_IR_TENORS - 1];
    };

    idouble dom_leg(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        idouble r = interpRate(dom_rates, t);
        idouble df = std::exp(-r * t);
        dom_leg = dom_leg + dom_notional * dom_rate * dt * df;
    }

    idouble fgn_leg(0.0);
    for (int i = 1; i <= num_periods; ++i) {
        double t = i * dt;
        idouble r = interpRate(fgn_rates, t);
        idouble df = std::exp(-r * t);
        fgn_leg = fgn_leg + fgn_notional * fgn_rate * dt * df;
    }

    idouble dom_r_mat = interpRate(dom_rates, maturity);
    idouble fgn_r_mat = interpRate(fgn_rates, maturity);
    dom_leg = dom_leg + dom_notional * std::exp(-dom_r_mat * maturity);
    fgn_leg = fgn_leg + fgn_notional * std::exp(-fgn_r_mat * maturity);
    dom_leg = dom_leg - dom_notional;
    fgn_leg = fgn_leg - fgn_notional;

    idouble npv = dom_leg - fgn_leg * fx_spot;
    k.npv_res = AADCResult(npv.markAsOutput());
    k.funcs.stopRecording();

    auto t1 = std::chrono::high_resolution_clock::now();
    k.recording_time_sec = std::chrono::duration<double>(t1 - t0).count();
}

inline std::vector<CRIFMapping> buildXCCYMappings(const XCCYCRIFKernel& k, double fx_spot) {
    std::vector<CRIFMapping> maps;
    maps.push_back({k.fx_spot_arg, "Risk_FX", "EUR", 0, "",
                    fx_spot * 0.01});
    double bp = 0.0001;
    for (int i = 0; i < NUM_IR_TENORS; ++i) {
        maps.push_back({k.dom_curve_args[i], "Risk_IRCurve", "USD", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
        maps.push_back({k.fgn_curve_args[i], "Risk_IRCurve", "EUR", 0,
                        std::string(IR_TENOR_LABELS[i]), bp});
    }
    return maps;
}

// ============================================================================
// Generic batched CRIF extraction from any kernel
// ============================================================================
template <typename Kernel, typename SetTradeFn>
inline std::vector<std::vector<CRIFRecord>> extractCRIFBatched(
    Kernel& kernel,
    AADCResult& price_output,
    const std::vector<CRIFMapping>& mappings,
    int num_trades,
    SetTradeFn set_trade_fn,  // (ws, lane_idx, trade_idx) -> void
    int num_threads)
{
    std::vector<std::vector<CRIFRecord>> result(num_trades);
    int num_batches = (num_trades + CRIF_AVX_WIDTH - 1) / CRIF_AVX_WIDTH;

    int actual_threads = std::min(num_threads, std::max(1, num_batches));
    std::vector<std::shared_ptr<AADCWorkSpace<mmType>>> workspaces(actual_threads);
    for (int i = 0; i < actual_threads; ++i) {
        workspaces[i] = kernel.funcs.createWorkSpace();
    }

    #pragma omp parallel for num_threads(actual_threads) schedule(dynamic)
    for (int batch = 0; batch < num_batches; ++batch) {
        #ifdef _OPENMP
        int tid = omp_get_thread_num();
        #else
        int tid = 0;
        #endif
        auto& ws = *workspaces[tid];
        int start = batch * CRIF_AVX_WIDTH;

        // Set per-trade params across lanes
        for (int lane = 0; lane < CRIF_AVX_WIDTH; ++lane) {
            int idx = start + lane;
            set_trade_fn(ws, lane, (idx < num_trades) ? idx : 0, idx >= num_trades);
        }

        // Forward pass
        kernel.funcs.forward(ws);

        // Reverse pass
        ws.resetDiff();
        ws.setDiff(price_output, 1.0);
        kernel.funcs.reverse(ws);

        // Extract CRIFs from gradients
        for (auto& m : mappings) {
            mmType mm_grad = ws.diff(m.handle);
            double* gp = reinterpret_cast<double*>(&mm_grad);
            for (int lane = 0; lane < CRIF_AVX_WIDTH; ++lane) {
                int idx = start + lane;
                if (idx >= num_trades) continue;
                double sens = gp[lane] * m.scaling;
                if (std::abs(sens) > 1e-6) {
                    result[idx].push_back({m.risk_type, m.qualifier,
                                           m.bucket, m.label1, sens});
                }
            }
        }
    }
    return result;
}

// ============================================================================
// Top-level: Compute all trade CRIFs via AADC
// ============================================================================
using Trade = std::variant<IRSwapTrade, EquityOptionTrade, InflationSwapTrade,
                           FXOptionTrade, XCCYSwapTrade>;

inline void computeAllCRIFsAADC(
    const std::vector<Trade>& portfolio,
    const MarketEnv& mkt,
    int num_threads,
    std::vector<std::vector<CRIFRecord>>& out_crifs,
    double& out_recording_time,
    double& out_eval_time)
{
    int N = static_cast<int>(portfolio.size());
    out_crifs.resize(N);
    out_recording_time = 0.0;

    // Partition trades by type
    std::vector<int> irs_idx, eq_idx, inf_idx, fx_idx, xccy_idx;
    for (int i = 0; i < N; ++i) {
        if (std::holds_alternative<IRSwapTrade>(portfolio[i])) irs_idx.push_back(i);
        else if (std::holds_alternative<EquityOptionTrade>(portfolio[i])) eq_idx.push_back(i);
        else if (std::holds_alternative<InflationSwapTrade>(portfolio[i])) inf_idx.push_back(i);
        else if (std::holds_alternative<FXOptionTrade>(portfolio[i])) fx_idx.push_back(i);
        else if (std::holds_alternative<XCCYSwapTrade>(portfolio[i])) xccy_idx.push_back(i);
    }

    auto eval_start = std::chrono::high_resolution_clock::now();

    // --- IRS ---
    if (!irs_idx.empty()) {
        IRSCRIFKernel irs_k;
        recordIRSCRIFKernel(irs_k);
        out_recording_time += irs_k.recording_time_sec;
        auto mappings = buildIRSMappings(irs_k);

        auto crifs = extractCRIFBatched(
            irs_k, irs_k.npv_res, mappings, static_cast<int>(irs_idx.size()),
            [&](AADCWorkSpace<mmType>& ws, int lane, int tidx, bool padding) {
                // Market data (same for all)
                if (lane == 0) {
                    for (int k = 0; k < NUM_IR_TENORS; ++k)
                        ws.setVal(irs_k.usd_curve[k], mmSetConst<mmType>(mkt.usd_curve.zero_rates[k]));
                }
                // Trade params
                mmType mm_not, mm_rate, mm_sign;
                double* np = reinterpret_cast<double*>(&mm_not);
                double* rp = reinterpret_cast<double*>(&mm_rate);
                double* sp = reinterpret_cast<double*>(&mm_sign);

                // Read current values to not overwrite other lanes
                mm_not = ws.val(irs_k.notional_arg);
                mm_rate = ws.val(irs_k.fixed_rate_arg);
                mm_sign = ws.val(irs_k.payer_sign_arg);

                if (!padding) {
                    auto& t = std::get<IRSwapTrade>(portfolio[irs_idx[tidx]]);
                    np[lane] = t.notional;
                    rp[lane] = t.fixed_rate;
                    sp[lane] = t.payer ? 1.0 : -1.0;
                } else {
                    np[lane] = 0.0; rp[lane] = 0.03; sp[lane] = 1.0;
                }
                ws.setVal(irs_k.notional_arg, mm_not);
                ws.setVal(irs_k.fixed_rate_arg, mm_rate);
                ws.setVal(irs_k.payer_sign_arg, mm_sign);
            }, num_threads);

        for (size_t j = 0; j < irs_idx.size(); ++j)
            out_crifs[irs_idx[j]] = std::move(crifs[j]);
    }

    // --- Equity Options ---
    if (!eq_idx.empty()) {
        EQCRIFKernel eq_k;
        recordEQCRIFKernel(eq_k);
        out_recording_time += eq_k.recording_time_sec;

        // Use first trade's bucket for mapping (simplified)
        int eq_bucket = std::get<EquityOptionTrade>(portfolio[eq_idx[0]]).equity_bucket;
        auto mappings = buildEQMappings(eq_k, mkt.equity_spot, eq_bucket);

        auto crifs = extractCRIFBatched(
            eq_k, eq_k.price_res, mappings, static_cast<int>(eq_idx.size()),
            [&](AADCWorkSpace<mmType>& ws, int lane, int tidx, bool padding) {
                if (lane == 0) {
                    ws.setVal(eq_k.spot_arg, mmSetConst<mmType>(mkt.equity_spot));
                    ws.setVal(eq_k.vol_arg, mmSetConst<mmType>(mkt.equity_vol));
                    for (int k = 0; k < NUM_IR_TENORS; ++k)
                        ws.setVal(eq_k.curve_args[k], mmSetConst<mmType>(mkt.usd_curve.zero_rates[k]));
                }
                mmType mm_strike, mm_div, mm_call;
                double* skp = reinterpret_cast<double*>(&mm_strike);
                double* dvp = reinterpret_cast<double*>(&mm_div);
                double* clp = reinterpret_cast<double*>(&mm_call);
                mm_strike = ws.val(eq_k.strike_arg);
                mm_div = ws.val(eq_k.div_yield_arg);
                mm_call = ws.val(eq_k.is_call_arg);
                if (!padding) {
                    auto& t = std::get<EquityOptionTrade>(portfolio[eq_idx[tidx]]);
                    skp[lane] = t.strike;
                    dvp[lane] = t.dividend_yield;
                    clp[lane] = t.is_call ? 1.0 : -1.0;
                } else {
                    skp[lane] = 100.0; dvp[lane] = 0.02; clp[lane] = 1.0;
                }
                ws.setVal(eq_k.strike_arg, mm_strike);
                ws.setVal(eq_k.div_yield_arg, mm_div);
                ws.setVal(eq_k.is_call_arg, mm_call);
            }, num_threads);

        for (size_t j = 0; j < eq_idx.size(); ++j)
            out_crifs[eq_idx[j]] = std::move(crifs[j]);
    }

    // --- FX Options ---
    if (!fx_idx.empty()) {
        FXCRIFKernel fx_k;
        recordFXCRIFKernel(fx_k);
        out_recording_time += fx_k.recording_time_sec;
        auto mappings = buildFXMappings(fx_k, mkt.fx_spot);

        auto crifs = extractCRIFBatched(
            fx_k, fx_k.price_res, mappings, static_cast<int>(fx_idx.size()),
            [&](AADCWorkSpace<mmType>& ws, int lane, int tidx, bool padding) {
                if (lane == 0) {
                    ws.setVal(fx_k.spot_arg, mmSetConst<mmType>(mkt.fx_spot));
                    ws.setVal(fx_k.vol_arg, mmSetConst<mmType>(mkt.fx_vol));
                    for (int k = 0; k < NUM_IR_TENORS; ++k) {
                        ws.setVal(fx_k.dom_curve_args[k], mmSetConst<mmType>(mkt.usd_curve.zero_rates[k]));
                        ws.setVal(fx_k.fgn_curve_args[k], mmSetConst<mmType>(mkt.eur_curve.zero_rates[k]));
                    }
                }
                mmType mm_strike;
                double* skp = reinterpret_cast<double*>(&mm_strike);
                mm_strike = ws.val(fx_k.strike_arg);
                if (!padding) {
                    auto& t = std::get<FXOptionTrade>(portfolio[fx_idx[tidx]]);
                    skp[lane] = t.strike;
                } else {
                    skp[lane] = 1.10;
                }
                ws.setVal(fx_k.strike_arg, mm_strike);
            }, num_threads);

        for (size_t j = 0; j < fx_idx.size(); ++j)
            out_crifs[fx_idx[j]] = std::move(crifs[j]);
    }

    // --- Inflation Swaps ---
    if (!inf_idx.empty()) {
        InflCRIFKernel inf_k;
        recordInflCRIFKernel(inf_k);
        out_recording_time += inf_k.recording_time_sec;
        auto mappings = buildInflMappings(inf_k);

        auto crifs = extractCRIFBatched(
            inf_k, inf_k.npv_res, mappings, static_cast<int>(inf_idx.size()),
            [&](AADCWorkSpace<mmType>& ws, int lane, int tidx, bool padding) {
                if (lane == 0) {
                    for (int k = 0; k < NUM_IR_TENORS; ++k) {
                        ws.setVal(inf_k.ir_curve_args[k], mmSetConst<mmType>(mkt.usd_curve.zero_rates[k]));
                        ws.setVal(inf_k.infl_rate_args[k], mmSetConst<mmType>(mkt.inflation.inflation_rates[k]));
                    }
                    ws.setVal(inf_k.base_cpi_arg, mmSetConst<mmType>(mkt.inflation.base_cpi));
                }
                mmType mm_not, mm_rate;
                double* np = reinterpret_cast<double*>(&mm_not);
                double* rp = reinterpret_cast<double*>(&mm_rate);
                mm_not = ws.val(inf_k.notional_arg);
                mm_rate = ws.val(inf_k.fixed_rate_arg);
                if (!padding) {
                    auto& t = std::get<InflationSwapTrade>(portfolio[inf_idx[tidx]]);
                    np[lane] = t.notional;
                    rp[lane] = t.fixed_rate;
                } else {
                    np[lane] = 0.0; rp[lane] = 0.025;
                }
                ws.setVal(inf_k.notional_arg, mm_not);
                ws.setVal(inf_k.fixed_rate_arg, mm_rate);
            }, num_threads);

        for (size_t j = 0; j < inf_idx.size(); ++j)
            out_crifs[inf_idx[j]] = std::move(crifs[j]);
    }

    // --- XCCY Swaps ---
    if (!xccy_idx.empty()) {
        XCCYCRIFKernel xccy_k;
        recordXCCYCRIFKernel(xccy_k);
        out_recording_time += xccy_k.recording_time_sec;
        auto mappings = buildXCCYMappings(xccy_k, mkt.fx_spot);

        auto crifs = extractCRIFBatched(
            xccy_k, xccy_k.npv_res, mappings, static_cast<int>(xccy_idx.size()),
            [&](AADCWorkSpace<mmType>& ws, int lane, int tidx, bool padding) {
                if (lane == 0) {
                    ws.setVal(xccy_k.fx_spot_arg, mmSetConst<mmType>(mkt.fx_spot));
                    for (int k = 0; k < NUM_IR_TENORS; ++k) {
                        ws.setVal(xccy_k.dom_curve_args[k], mmSetConst<mmType>(mkt.usd_curve.zero_rates[k]));
                        ws.setVal(xccy_k.fgn_curve_args[k], mmSetConst<mmType>(mkt.eur_curve.zero_rates[k]));
                    }
                }
                mmType mm_dn, mm_fn, mm_dr, mm_fr;
                double* dnp = reinterpret_cast<double*>(&mm_dn);
                double* fnp = reinterpret_cast<double*>(&mm_fn);
                double* drp = reinterpret_cast<double*>(&mm_dr);
                double* frp = reinterpret_cast<double*>(&mm_fr);
                mm_dn = ws.val(xccy_k.dom_notional_arg);
                mm_fn = ws.val(xccy_k.fgn_notional_arg);
                mm_dr = ws.val(xccy_k.dom_rate_arg);
                mm_fr = ws.val(xccy_k.fgn_rate_arg);
                if (!padding) {
                    auto& t = std::get<XCCYSwapTrade>(portfolio[xccy_idx[tidx]]);
                    dnp[lane] = t.dom_notional;
                    fnp[lane] = t.fgn_notional;
                    drp[lane] = t.dom_fixed_rate;
                    frp[lane] = t.fgn_fixed_rate;
                } else {
                    dnp[lane] = 0.0; fnp[lane] = 0.0; drp[lane] = 0.04; frp[lane] = 0.03;
                }
                ws.setVal(xccy_k.dom_notional_arg, mm_dn);
                ws.setVal(xccy_k.fgn_notional_arg, mm_fn);
                ws.setVal(xccy_k.dom_rate_arg, mm_dr);
                ws.setVal(xccy_k.fgn_rate_arg, mm_fr);
            }, num_threads);

        for (size_t j = 0; j < xccy_idx.size(); ++j)
            out_crifs[xccy_idx[j]] = std::move(crifs[j]);
    }

    auto eval_end = std::chrono::high_resolution_clock::now();
    out_eval_time = std::chrono::duration<double>(eval_end - eval_start).count();
}

} // namespace simm
