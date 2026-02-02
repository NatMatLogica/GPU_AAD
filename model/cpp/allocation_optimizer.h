// Allocation Optimizer with Batched AADC Evaluation
// Ported from model/simm_portfolio_aadc_v2.py + simm_allocation_optimizer_v2.py
// Version: 2.1.0
#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <aadc/aadc.h>
#include <aadc/aadc_matrix.h>

#include "simm_aggregation.h"
#include "sensitivity_matrix.h"

using aadc::AADCWorkSpace;

namespace simm {

// ============================================================================
// Allocation Matrix (row-major T x P)
// ============================================================================
struct AllocationMatrix {
    std::vector<double> data;
    int T, P;

    AllocationMatrix() : T(0), P(0) {}
    AllocationMatrix(int T_, int P_) : data(T_ * P_, 0.0), T(T_), P(P_) {}

    double& operator()(int t, int p) { return data[t * P + p]; }
    double operator()(int t, int p) const { return data[t * P + p]; }

    AllocationMatrix copy() const {
        AllocationMatrix c;
        c.data = data;
        c.T = T;
        c.P = P;
        return c;
    }
};

// ============================================================================
// Evaluation Result
// ============================================================================
struct EvalResult {
    std::vector<double> gradient;   // T x P, row-major
    std::vector<double> ims;        // P values
    double eval_time_sec;
};

// ============================================================================
// Optimization Result
// ============================================================================
struct OptimizationResult {
    AllocationMatrix final_allocation;
    double initial_im;
    double final_im;
    std::vector<double> im_history;
    int num_iterations;
    int trades_moved;
    double total_eval_time_sec;
    double kernel_recording_time_sec;
    int total_evals = 0;           // Total SIMM evaluation count (Phase 1 + greedy)
    int greedy_rounds = 0;         // Number of greedy refinement rounds
};

// ============================================================================
// Matrix Multiply Helpers (OpenMP-parallel)
// ============================================================================

// C = A^T @ B where A is (M,K) row-major, B is (M,N) row-major -> C is (K,N)
// Accumulation-based: iterate rows of A and B together for cache-friendly access.
inline void matmulATB(const double* A, const double* B,
                      double* C, int M, int K, int N) {
    std::fill(C, C + K * N, 0.0);
    // For small N (portfolios), parallelize over K and accumulate across M rows.
    // Each thread accumulates partial sums to avoid false sharing.
    #pragma omp parallel
    {
        std::vector<double> local(K * N, 0.0);
        #pragma omp for schedule(static)
        for (int m = 0; m < M; ++m) {
            const double* a_row = A + m * K;
            const double* b_row = B + m * N;
            for (int k = 0; k < K; ++k) {
                double a_val = a_row[k];
                for (int n = 0; n < N; ++n) {
                    local[k * N + n] += a_val * b_row[n];
                }
            }
        }
        #pragma omp critical
        for (int i = 0; i < K * N; ++i) C[i] += local[i];
    }
}

// C = A @ B where A is (M,K) row-major, B is (K,N) row-major -> C is (M,N)
// Row-of-A dot columns-of-B with accumulation for better cache reuse on B.
inline void matmulAB(const double* A, const double* B,
                     double* C, int M, int K, int N) {
    std::fill(C, C + M * N, 0.0);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const double* a_row = A + m * K;
        double* c_row = C + m * N;
        for (int k = 0; k < K; ++k) {
            double a_val = a_row[k];
            const double* b_row = B + k * N;
            for (int n = 0; n < N; ++n) {
                c_row[n] += a_val * b_row[n];
            }
        }
    }
}

// ============================================================================
// Core: Evaluate ALL P portfolios + gradients in ONE AADC call
// Port of compute_all_portfolios_im_gradient_v2
// ============================================================================
inline EvalResult evaluateAllPortfolios(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    const AllocationMatrix& allocation,
    int num_threads)
{
    int T = allocation.T;
    int P = allocation.P;
    int K = kernel.K;
    assert(S.T == T && S.K == K);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: agg_S = S^T @ allocation -> (K, P)
    std::vector<double> agg_S(K * P);
    matmulATB(S.data.data(), allocation.data.data(), agg_S.data(), T, K, P);

    // Step 2: Evaluate AADC kernel for P portfolios in batches of AVX_WIDTH
    auto ws = kernel.funcs.createWorkSpace();
    int num_batches = (P + AVX_WIDTH - 1) / AVX_WIDTH;

    std::vector<double> ims(P, 0.0);
    std::vector<double> grad_matrix(K * P, 0.0);  // K x P

    for (int batch = 0; batch < num_batches; ++batch) {
        int p_start = batch * AVX_WIDTH;

        // Set K inputs, each as __m256d with up to 4 portfolio values
        for (int k = 0; k < K; ++k) {
            mmType mm_val;
            double* ptr = reinterpret_cast<double*>(&mm_val);
            for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                int p = p_start + lane;
                ptr[lane] = (p < P) ? agg_S[k * P + p] : 0.0;
            }
            ws->setVal(kernel.sens_handles[k], mm_val);
        }

        // Forward pass
        kernel.funcs.forward(*ws);

        // Extract IM values
        {
            mmType mm_im = ws->val(kernel.im_output);
            double* im_ptr = reinterpret_cast<double*>(&mm_im);
            for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                int p = p_start + lane;
                if (p < P) ims[p] = im_ptr[lane];
            }
        }

        // Reverse pass for gradients
        ws->resetDiff();
        ws->setDiff(kernel.im_output, 1.0);
        kernel.funcs.reverse(*ws);

        // Extract K gradients for these portfolios
        for (int k = 0; k < K; ++k) {
            mmType mm_grad = ws->diff(kernel.sens_handles[k]);
            double* grad_ptr = reinterpret_cast<double*>(&mm_grad);
            for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                int p = p_start + lane;
                if (p < P) grad_matrix[k * P + p] = grad_ptr[lane];
            }
        }
    }

    // Step 3: Chain rule: gradient[T,P] = S[T,K] @ grad_matrix[K,P]
    std::vector<double> gradient(T * P);
    matmulAB(S.data.data(), grad_matrix.data(), gradient.data(), T, K, P);

    auto t_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(t_end - t_start).count();

    return {std::move(gradient), std::move(ims), eval_time};
}

// ============================================================================
// Multi-threaded evaluation with multiple workspaces
// ============================================================================
inline EvalResult evaluateAllPortfoliosMT(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    const AllocationMatrix& allocation,
    int num_threads)
{
    int T = allocation.T;
    int P = allocation.P;
    int K = kernel.K;

    if (P <= AVX_WIDTH || num_threads <= 1) {
        return evaluateAllPortfolios(kernel, S, allocation, num_threads);
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: agg_S = S^T @ allocation -> (K, P)
    std::vector<double> agg_S(K * P);
    matmulATB(S.data.data(), allocation.data.data(), agg_S.data(), T, K, P);

    // Step 2: Parallel AADC evaluation
    int num_batches = (P + AVX_WIDTH - 1) / AVX_WIDTH;
    std::vector<double> ims(P, 0.0);
    std::vector<double> grad_matrix(K * P, 0.0);

    // Create one workspace per thread
    int actual_threads = std::min(num_threads, num_batches);
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

        int p_start = batch * AVX_WIDTH;

        for (int k = 0; k < K; ++k) {
            mmType mm_val;
            double* ptr = reinterpret_cast<double*>(&mm_val);
            for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                int p = p_start + lane;
                ptr[lane] = (p < P) ? agg_S[k * P + p] : 0.0;
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
                if (p < P) grad_matrix[k * P + p] = grad_ptr[lane];
            }
        }
    }

    // Step 3: Chain rule
    std::vector<double> gradient(T * P);
    matmulAB(S.data.data(), grad_matrix.data(), gradient.data(), T, K, P);

    auto t_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(t_end - t_start).count();

    return {std::move(gradient), std::move(ims), eval_time};
}

// ============================================================================
// Pre-Weighted Evaluation: WS = S * diag(rw*cr) aggregated outside kernel
// Gradients: dIM/ds = dIM/dWS * rw * cr, applied via chain rule on weighted_S
// ============================================================================
inline EvalResult evaluateAllPortfoliosPreWeightedMT(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    const AllocationMatrix& allocation,
    const std::vector<FactorMeta>& metadata,
    int num_threads)
{
    int T = allocation.T;
    int P = allocation.P;
    int K = kernel.K;
    assert(S.T == T && S.K == K);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: Pre-weight sensitivities: weighted_S[t,k] = S[t,k] * rw[k] * cr[k]
    std::vector<double> weighted_S(T * K);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
            weighted_S[t * K + k] = S.data[t * K + k] * metadata[k].weight * metadata[k].cr;

    // Step 2: agg_WS = weighted_S^T @ allocation -> (K, P)
    std::vector<double> agg_WS(K * P);
    matmulATB(weighted_S.data(), allocation.data.data(), agg_WS.data(), T, K, P);

    // Step 3: Evaluate AADC kernel (same batched AVX logic)
    int num_batches = (P + AVX_WIDTH - 1) / AVX_WIDTH;
    std::vector<double> ims(P, 0.0);
    std::vector<double> grad_matrix(K * P, 0.0);

    if (P <= AVX_WIDTH || num_threads <= 1) {
        auto ws = kernel.funcs.createWorkSpace();
        for (int batch = 0; batch < num_batches; ++batch) {
            int p_start = batch * AVX_WIDTH;
            for (int k = 0; k < K; ++k) {
                mmType mm_val;
                double* ptr = reinterpret_cast<double*>(&mm_val);
                for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                    int p = p_start + lane;
                    ptr[lane] = (p < P) ? agg_WS[k * P + p] : 0.0;
                }
                ws->setVal(kernel.sens_handles[k], mm_val);
            }
            kernel.funcs.forward(*ws);
            {
                mmType mm_im = ws->val(kernel.im_output);
                double* ip = reinterpret_cast<double*>(&mm_im);
                for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                    int p = p_start + lane;
                    if (p < P) ims[p] = ip[lane];
                }
            }
            ws->resetDiff();
            ws->setDiff(kernel.im_output, 1.0);
            kernel.funcs.reverse(*ws);
            for (int k = 0; k < K; ++k) {
                mmType mm_grad = ws->diff(kernel.sens_handles[k]);
                double* gp = reinterpret_cast<double*>(&mm_grad);
                for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                    int p = p_start + lane;
                    if (p < P) grad_matrix[k * P + p] = gp[lane];
                }
            }
        }
    } else {
        int actual_threads = std::min(num_threads, num_batches);
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
            int p_start = batch * AVX_WIDTH;
            for (int k = 0; k < K; ++k) {
                mmType mm_val;
                double* ptr = reinterpret_cast<double*>(&mm_val);
                for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                    int p = p_start + lane;
                    ptr[lane] = (p < P) ? agg_WS[k * P + p] : 0.0;
                }
                ws.setVal(kernel.sens_handles[k], mm_val);
            }
            kernel.funcs.forward(ws);
            {
                mmType mm_im = ws.val(kernel.im_output);
                double* ip = reinterpret_cast<double*>(&mm_im);
                for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                    int p = p_start + lane;
                    if (p < P) ims[p] = ip[lane];
                }
            }
            ws.resetDiff();
            ws.setDiff(kernel.im_output, 1.0);
            kernel.funcs.reverse(ws);
            for (int k = 0; k < K; ++k) {
                mmType mm_grad = ws.diff(kernel.sens_handles[k]);
                double* gp = reinterpret_cast<double*>(&mm_grad);
                for (int lane = 0; lane < AVX_WIDTH; ++lane) {
                    int p = p_start + lane;
                    if (p < P) grad_matrix[k * P + p] = gp[lane];
                }
            }
        }
    }

    // Step 4: Chain rule with weighted_S (not S)
    // gradient[t,p] = sum_k weighted_S[t,k] * grad_matrix[k,p]
    std::vector<double> gradient(T * P);
    matmulAB(weighted_S.data(), grad_matrix.data(), gradient.data(), T, K, P);

    auto t_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(t_end - t_start).count();

    return {std::move(gradient), std::move(ims), eval_time};
}

// ============================================================================
// Forward-only evaluation from pre-computed agg_S (no chain-rule matmuls).
// Returns IM values only. Used by incremental greedy search.
// ============================================================================
inline std::vector<double> evaluateIMFromAggS(
    SIMMKernel& kernel,
    const double* agg_S,   // K x P, column-major (K rows, P cols)
    int P,
    int num_threads)
{
    int K = kernel.K;
    int num_batches = (P + AVX_WIDTH - 1) / AVX_WIDTH;
    std::vector<double> ims(P, 0.0);

    int actual_threads = std::min(num_threads, num_batches);
    std::vector<std::shared_ptr<AADCWorkSpace<mmType>>> workspaces(actual_threads);
    for (int i = 0; i < actual_threads; ++i)
        workspaces[i] = kernel.funcs.createWorkSpace();

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
                ptr[lane] = (p < P) ? agg_S[k * P + p] : 0.0;
            }
            ws.setVal(kernel.sens_handles[k], mm_val);
        }

        kernel.funcs.forward(ws);

        mmType mm_im = ws.val(kernel.im_output);
        double* im_ptr = reinterpret_cast<double*>(&mm_im);
        for (int lane = 0; lane < AVX_WIDTH; ++lane) {
            int p = p_start + lane;
            if (p < P) ims[p] = im_ptr[lane];
        }
    }
    return ims;
}

// ============================================================================
// Simplex Projection (Duchi et al.)
// Projects each row to probability simplex: sum=1, all>=0
// ============================================================================
inline void projectToSimplex(AllocationMatrix& x) {
    #pragma omp parallel for
    for (int t = 0; t < x.T; ++t) {
        int P = x.P;
        // Copy and sort descending
        std::vector<double> u(P);
        for (int p = 0; p < P; ++p) u[p] = x(t, p);
        std::sort(u.begin(), u.end(), std::greater<double>());

        // Find rho
        double cssv = 0.0;
        int rho = -1;
        for (int j = 0; j < P; ++j) {
            cssv += u[j];
            if (u[j] * (j + 1) > cssv - 1.0) rho = j;
        }

        // Compute theta and project
        if (rho >= 0) {
            double sum = 0.0;
            for (int j = 0; j <= rho; ++j) sum += u[j];
            double theta = (sum - 1.0) / (rho + 1);
            for (int p = 0; p < P; ++p) {
                x(t, p) = std::max(x(t, p) - theta, 0.0);
            }
        } else {
            // Uniform
            for (int p = 0; p < P; ++p) x(t, p) = 1.0 / P;
        }
    }
}

// ============================================================================
// Round continuous allocation to integer (argmax per row)
// ============================================================================
inline AllocationMatrix roundToInteger(const AllocationMatrix& x) {
    AllocationMatrix result(x.T, x.P);
    for (int t = 0; t < x.T; ++t) {
        int best_p = 0;
        double best_val = x(t, 0);
        for (int p = 1; p < x.P; ++p) {
            if (x(t, p) > best_val) {
                best_val = x(t, p);
                best_p = p;
            }
        }
        result(t, best_p) = 1.0;
    }
    return result;
}

// ============================================================================
// Count trades moved vs initial allocation
// ============================================================================
inline int countTradesMoved(const AllocationMatrix& initial, const AllocationMatrix& final_alloc) {
    int moved = 0;
    for (int t = 0; t < initial.T; ++t) {
        int init_p = 0, final_p = 0;
        double init_max = initial(t, 0), final_max = final_alloc(t, 0);
        for (int p = 1; p < initial.P; ++p) {
            if (initial(t, p) > init_max) { init_max = initial(t, p); init_p = p; }
            if (final_alloc(t, p) > final_max) { final_max = final_alloc(t, p); final_p = p; }
        }
        if (init_p != final_p) ++moved;
    }
    return moved;
}

// ============================================================================
// Gradient Descent with Armijo Line Search
// ============================================================================
inline OptimizationResult optimizeGradientDescent(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    AllocationMatrix initial_allocation,
    int num_threads,
    int max_iters = 100,
    double lr_init = 0.0,
    double tol = 1e-6,
    bool verbose = true)
{
    constexpr double LS_BETA = 0.5;
    constexpr int LS_MAX_TRIES = 10;

    AllocationMatrix x = initial_allocation.copy();
    AllocationMatrix best_x = x.copy();
    std::vector<double> im_history;
    double total_eval_time = 0.0;

    // First evaluation
    auto result = evaluateAllPortfoliosMT(kernel, S, x, num_threads);
    double total_im = std::accumulate(result.ims.begin(), result.ims.end(), 0.0);
    im_history.push_back(total_im);
    total_eval_time += result.eval_time_sec;
    double best_im = total_im;
    double initial_im = total_im;

    // Auto learning rate
    double lr = lr_init;
    if (lr <= 0.0) {
        double grad_max = 0.0;
        for (double g : result.gradient) grad_max = std::max(grad_max, std::abs(g));
        lr = (grad_max > 1e-10) ? 1.0 / grad_max : 1e-12;
    }

    if (verbose) {
        std::cout << "    [C++ GD] Initial IM: $" << std::fixed << std::setprecision(2) << total_im << "\n";
        std::cout << "    [C++ GD] Learning rate: " << std::scientific << lr << "\n";
        std::cout << "    [C++ GD] Eval time: " << std::fixed << std::setprecision(2)
                  << result.eval_time_sec * 1000 << " ms\n";
    }

    int stalled = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        if (iter > 0) {
            result = evaluateAllPortfoliosMT(kernel, S, x, num_threads);
            total_im = std::accumulate(result.ims.begin(), result.ims.end(), 0.0);
            im_history.push_back(total_im);
            total_eval_time += result.eval_time_sec;
        }

        if (total_im < best_im) {
            best_im = total_im;
            best_x = x.copy();
            stalled = 0;
        } else {
            ++stalled;
        }

        if (verbose && iter % 10 == 0) {
            int moves = countTradesMoved(initial_allocation, x);
            std::cout << "    [C++ GD] Iter " << iter << ": IM = $" << std::fixed
                      << std::setprecision(2) << total_im << ", best = $" << best_im
                      << ", moves = " << moves << "\n";
        }

        if (stalled >= 20) {
            if (verbose) std::cout << "    [C++ GD] Stalled, reverting to best\n";
            x = best_x.copy();
            break;
        }

        // Backtracking line search
        double step_size = lr;
        for (int ls = 0; ls < LS_MAX_TRIES; ++ls) {
            AllocationMatrix x_cand = x.copy();
            for (int i = 0; i < x.T * x.P; ++i) {
                x_cand.data[i] -= step_size * result.gradient[i];
            }
            projectToSimplex(x_cand);

            auto cand_result = evaluateAllPortfoliosMT(kernel, S, x_cand, num_threads);
            double cand_im = std::accumulate(cand_result.ims.begin(), cand_result.ims.end(), 0.0);
            total_eval_time += cand_result.eval_time_sec;

            if (cand_im < total_im) {
                x = std::move(x_cand);
                break;
            }
            step_size *= LS_BETA;
        }

        // Convergence check
        if (im_history.size() >= 2) {
            double rel = std::abs(im_history.back() - im_history[im_history.size()-2]) /
                         std::max(std::abs(im_history[im_history.size()-2]), 1e-10);
            if (rel < tol) {
                if (verbose) std::cout << "    [C++ GD] Converged at iter " << iter+1 << "\n";
                break;
            }
        }
    }

    int trades_moved = countTradesMoved(initial_allocation, best_x);

    if (verbose) {
        std::cout << "    [C++ GD] Final IM: $" << std::fixed << std::setprecision(2) << best_im
                  << " (reduction: " << std::setprecision(1)
                  << 100.0 * (1.0 - best_im / initial_im) << "%)\n";
        std::cout << "    [C++ GD] Trades moved: " << trades_moved << "\n";
        std::cout << "    [C++ GD] Total eval time: " << std::setprecision(2)
                  << total_eval_time * 1000 << " ms\n";
    }

    int phase1_evals = static_cast<int>(im_history.size());  // 1 per iter + initial
    return {std::move(best_x), initial_im, best_im, std::move(im_history),
            static_cast<int>(im_history.size()), trades_moved,
            total_eval_time, kernel.recording_time_sec,
            phase1_evals, 0};
}

// ============================================================================
// Adam Optimizer
// ============================================================================
inline OptimizationResult optimizeAdam(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    AllocationMatrix initial_allocation,
    int num_threads,
    int max_iters = 100,
    double lr_init = 0.0,
    double tol = 1e-6,
    bool verbose = true,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double eps = 1e-8)
{
    constexpr double LS_BETA = 0.5;
    constexpr int LS_MAX_TRIES = 10;

    int TP = initial_allocation.T * initial_allocation.P;
    AllocationMatrix x = initial_allocation.copy();
    AllocationMatrix best_x = x.copy();
    std::vector<double> im_history;
    double total_eval_time = 0.0;

    // Adam moment vectors
    std::vector<double> m(TP, 0.0);
    std::vector<double> v(TP, 0.0);

    // First evaluation
    auto result = evaluateAllPortfoliosMT(kernel, S, x, num_threads);
    double total_im = std::accumulate(result.ims.begin(), result.ims.end(), 0.0);
    im_history.push_back(total_im);
    total_eval_time += result.eval_time_sec;
    double best_im = total_im;
    double initial_im = total_im;

    // Auto learning rate
    double lr = lr_init;
    if (lr <= 0.0) {
        double grad_max = 0.0;
        for (double g : result.gradient) grad_max = std::max(grad_max, std::abs(g));
        lr = (grad_max > 1e-10) ? 1.0 / grad_max : 1e-12;
    }

    if (verbose) {
        std::cout << "    [C++ Adam] Initial IM: $" << std::fixed << std::setprecision(2) << total_im << "\n";
        std::cout << "    [C++ Adam] Learning rate: " << std::scientific << lr << "\n";
    }

    int stalled = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        if (iter > 0) {
            result = evaluateAllPortfoliosMT(kernel, S, x, num_threads);
            total_im = std::accumulate(result.ims.begin(), result.ims.end(), 0.0);
            im_history.push_back(total_im);
            total_eval_time += result.eval_time_sec;
        }

        if (total_im < best_im) {
            best_im = total_im;
            best_x = x.copy();
            stalled = 0;
        } else {
            ++stalled;
        }

        if (verbose && iter % 10 == 0) {
            std::cout << "    [C++ Adam] Iter " << iter << ": IM = $" << std::fixed
                      << std::setprecision(2) << total_im << ", best = $" << best_im << "\n";
        }

        if (stalled >= 20) {
            x = best_x.copy();
            break;
        }

        // Adam update
        double t_adam = iter + 1;
        for (int i = 0; i < TP; ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * result.gradient[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * result.gradient[i] * result.gradient[i];
        }
        double bc1 = 1.0 - std::pow(beta1, t_adam);
        double bc2 = 1.0 - std::pow(beta2, t_adam);

        // Backtracking with Adam direction
        double step_size = lr;
        for (int ls = 0; ls < LS_MAX_TRIES; ++ls) {
            AllocationMatrix x_cand = x.copy();
            for (int i = 0; i < TP; ++i) {
                double m_hat = m[i] / bc1;
                double v_hat = v[i] / bc2;
                x_cand.data[i] -= step_size * m_hat / (std::sqrt(v_hat) + eps);
            }
            projectToSimplex(x_cand);

            auto cand_result = evaluateAllPortfoliosMT(kernel, S, x_cand, num_threads);
            double cand_im = std::accumulate(cand_result.ims.begin(), cand_result.ims.end(), 0.0);
            total_eval_time += cand_result.eval_time_sec;

            if (cand_im < total_im) {
                x = std::move(x_cand);
                break;
            }
            step_size *= LS_BETA;
        }

        // Convergence
        if (im_history.size() >= 2) {
            double rel = std::abs(im_history.back() - im_history[im_history.size()-2]) /
                         std::max(std::abs(im_history[im_history.size()-2]), 1e-10);
            if (rel < tol) {
                if (verbose) std::cout << "    [C++ Adam] Converged at iter " << iter+1 << "\n";
                break;
            }
        }
    }

    int trades_moved = countTradesMoved(initial_allocation, best_x);

    if (verbose) {
        std::cout << "    [C++ Adam] Final IM: $" << std::fixed << std::setprecision(2) << best_im
                  << " (reduction: " << std::setprecision(1)
                  << 100.0 * (1.0 - best_im / initial_im) << "%)\n";
        std::cout << "    [C++ Adam] Trades moved: " << trades_moved << "\n";
    }

    int phase1_evals = static_cast<int>(im_history.size());
    return {std::move(best_x), initial_im, best_im, std::move(im_history),
            static_cast<int>(im_history.size()), trades_moved,
            total_eval_time, kernel.recording_time_sec,
            phase1_evals, 0};
}

// ============================================================================
// Greedy Local Search (gradient-guided, incremental agg_S)
//
// Uses AADC gradients to rank candidate moves. For validation, maintains
// agg_S incrementally (O(K) per move) and runs forward-only AADC (no chain
// rule matmul). Full gradient evaluation only once per round for ranking.
// ============================================================================
inline OptimizationResult greedyLocalSearch(
    SIMMKernel& kernel,
    const SensitivityMatrix& S,
    AllocationMatrix integer_allocation,
    int num_threads,
    int max_rounds = 50,
    bool verbose = true)
{
    AllocationMatrix x = integer_allocation.copy();
    AllocationMatrix best_x = x.copy();
    std::vector<double> im_history;
    double total_eval_time = 0.0;
    int total_evals = 0;
    int T = x.T, P = x.P, K = kernel.K;

    // How many candidates to validate per round
    int top_k = std::min(T, std::max(20, T / 10));

    // Initial evaluation with full gradient (for ranking)
    auto result = evaluateAllPortfoliosMT(kernel, S, x, num_threads);
    double total_im = std::accumulate(result.ims.begin(), result.ims.end(), 0.0);
    im_history.push_back(total_im);
    total_eval_time += result.eval_time_sec;
    ++total_evals;
    double best_im = total_im;
    double initial_im = total_im;

    // Maintain running agg_S = S^T @ x -> (K, P)
    std::vector<double> agg_S(K * P);
    matmulATB(S.data.data(), x.data.data(), agg_S.data(), T, K, P);
    // Keep a copy of current IM per portfolio
    std::vector<double> port_ims = result.ims;

    if (verbose) {
        std::cout << "    [C++ Greedy] Initial IM: $" << std::fixed << std::setprecision(2)
                  << total_im << " (top_k=" << top_k << ")\n";
    }

    struct Candidate {
        int trade;
        int curr_p;
        int new_p;
        double predicted_delta;
    };

    for (int round = 0; round < max_rounds; ++round) {
        // Rank all possible moves using gradients
        std::vector<Candidate> candidates;
        candidates.reserve(T * (P - 1));

        for (int t = 0; t < T; ++t) {
            int curr_p = 0;
            for (int p = 1; p < P; ++p) {
                if (x(t, p) > x(t, curr_p)) curr_p = p;
            }
            double g_curr = result.gradient[t * P + curr_p];
            for (int new_p = 0; new_p < P; ++new_p) {
                if (new_p == curr_p) continue;
                double delta = result.gradient[t * P + new_p] - g_curr;
                if (delta < 0.0) {
                    candidates.push_back({t, curr_p, new_p, delta});
                }
            }
        }

        if (candidates.empty()) {
            if (verbose) std::cout << "    [C++ Greedy] No improving candidates, stopping at round "
                                   << round << "\n";
            break;
        }

        int n_try = std::min(top_k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(), candidates.begin() + n_try, candidates.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.predicted_delta < b.predicted_delta;
                          });

        bool improved = false;
        int accepted = 0;

        for (int i = 0; i < n_try; ++i) {
            auto& c = candidates[i];

            int actual_p = 0;
            for (int p = 1; p < P; ++p) {
                if (x(c.trade, p) > x(c.trade, actual_p)) actual_p = p;
            }
            if (actual_p != c.curr_p) continue;

            // Incremental agg_S update: O(K) instead of O(T*K*P) matmul
            for (int k = 0; k < K; ++k) {
                double s_tk = S.data[c.trade * K + k];
                agg_S[k * P + c.curr_p] -= s_tk;
                agg_S[k * P + c.new_p]  += s_tk;
            }

            // Forward-only evaluation from agg_S (no chain rule, no reverse)
            auto t_start = std::chrono::high_resolution_clock::now();
            auto trial_ims = evaluateIMFromAggS(kernel, agg_S.data(), P, num_threads);
            auto t_end = std::chrono::high_resolution_clock::now();
            total_eval_time += std::chrono::duration<double>(t_end - t_start).count();
            ++total_evals;

            double trial_im = std::accumulate(trial_ims.begin(), trial_ims.end(), 0.0);

            if (trial_im < total_im) {
                // Accept: update allocation and IM tracking
                x(c.trade, c.curr_p) = 0.0;
                x(c.trade, c.new_p) = 1.0;
                total_im = trial_im;
                port_ims = trial_ims;
                improved = true;
                ++accepted;
                if (total_im < best_im) {
                    best_im = total_im;
                    best_x = x.copy();
                }
            } else {
                // Revert agg_S
                for (int k = 0; k < K; ++k) {
                    double s_tk = S.data[c.trade * K + k];
                    agg_S[k * P + c.curr_p] += s_tk;
                    agg_S[k * P + c.new_p]  -= s_tk;
                }
            }
        }

        im_history.push_back(total_im);

        if (verbose) {
            std::cout << "    [C++ Greedy] Round " << round << ": IM = $" << std::fixed
                      << std::setprecision(2) << total_im << ", tried " << n_try
                      << ", accepted " << accepted << "\n";
        }

        if (!improved) {
            if (verbose) std::cout << "    [C++ Greedy] No improvement validated, stopping at round "
                                   << round << "\n";
            break;
        }

        // Refresh gradients for next round (full eval with chain rule)
        result = evaluateAllPortfoliosMT(kernel, S, x, num_threads);
        total_im = std::accumulate(result.ims.begin(), result.ims.end(), 0.0);
        total_eval_time += result.eval_time_sec;
        ++total_evals;
        port_ims = result.ims;
        // Recompute agg_S to stay in sync (avoids floating-point drift)
        matmulATB(S.data.data(), x.data.data(), agg_S.data(), T, K, P);
    }

    int trades_moved = countTradesMoved(integer_allocation, best_x);
    int greedy_rounds = static_cast<int>(im_history.size()) - 1;  // Exclude initial

    if (verbose) {
        std::cout << "    [C++ Greedy] Final IM: $" << std::fixed << std::setprecision(2) << best_im
                  << " (reduction: " << std::setprecision(1)
                  << 100.0 * (1.0 - best_im / initial_im) << "%, "
                  << total_evals << " evals, " << greedy_rounds << " rounds)\n";
    }

    return {std::move(best_x), initial_im, best_im, std::move(im_history),
            static_cast<int>(im_history.size()), trades_moved,
            total_eval_time, kernel.recording_time_sec,
            total_evals, greedy_rounds};
}

} // namespace simm
