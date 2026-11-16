// ==============================================================================
// DURBIN-WATSON MASSIVE SIMULATION - HIGH-PERFORMANCE NVIDIA CUDA C++ VERSION
// ==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "\n[CUDA ERROR] " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int SAMPLE_SIZE = 50;

// Store base sample in fast GPU constant memory
__constant__ double d_const_values[SAMPLE_SIZE];

struct SimStats {
    std::string label;
    long long n;
    double runtime_sec;
    double mean;
    double median;
    double min_val;
    double max_val;
    double q1;
    double q3;
    double sd;
};

// 1. GLOBAL PROGRESS BAR FUNCTION
void print_global_progress(const std::string& phase_name, int progress_percentage, const std::string& color_code) {
    const int bar_length = 30;
    int filled_length = static_cast<int>(std::round(bar_length * (progress_percentage / 100.0)));
    std::string bar;
    for (int i = 0; i < filled_length; ++i) bar += "█";
    for (int i = filled_length; i < bar_length; ++i) bar += "-";

    std::cout << color_code << "[GLOBAL PROGRESS] [" << bar << "] "
              << std::setw(3) << progress_percentage << "% | Phase: " << phase_name
              << "\033[0m\n" << std::flush;
}

// GPU Kernel: Each CUDA thread performs one Fisher-Yates shuffle + Durbin-Watson calculation
__global__ void durbin_watson_kernel(
    double* __restrict__ d_out,
    long long count,
    uint64_t seed,
    uint64_t sequence_offset,
    double inv_denom
) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    // Initialize Philox parallel PRNG deterministically for this thread
    curandStatePhilox4_32_10_t state;
    curand_init(seed, sequence_offset + static_cast<uint64_t>(tid), 0ULL, &state);

    // Load sample into fast thread-local array
    double local_buf[SAMPLE_SIZE];
    #pragma unroll
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        local_buf[i] = d_const_values[i];
    }

    // In-place Fisher-Yates shuffle
    for (int i = SAMPLE_SIZE - 1; i > 0; --i) {
        unsigned int r = curand(&state);
        int j = static_cast<int>(r % static_cast<unsigned int>(i + 1));
        double tmp = local_buf[i];
        local_buf[i] = local_buf[j];
        local_buf[j] = tmp;
    }

    // Compute Durbin-Watson numerator
    double num = 0.0;
    #pragma unroll
    for (int j = 0; j < SAMPLE_SIZE - 1; ++j) {
        double d = local_buf[j + 1] - local_buf[j];
        num += d * d;
    }

    d_out[tid] = num * inv_denom;
}

// Fast O(N) CPU quantile computation using std::nth_element
double compute_quantile(std::vector<double>& data, double q) {
    size_t n = data.size();
    double pos = q * (n - 1);
    size_t idx = static_cast<size_t>(pos);
    double frac = pos - idx;

    std::nth_element(data.begin(), data.begin() + idx, data.end());
    double val1 = data[idx];

    if (frac == 0.0 || idx + 1 >= n) {
        return val1;
    }
    double val2 = *std::min_element(data.begin() + idx + 1, data.end());
    return val1 + frac * (val2 - val1);
}

int main() {
    const std::string CYAN   = "\033[96m";
    const std::string YELLOW = "\033[93m";
    const std::string GREEN  = "\033[92m";

    auto global_start = std::chrono::high_resolution_clock::now();

    // Query GPU properties
    int device_id = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    print_global_progress("Initialization", 0, CYAN);
    std::cout << "Global timer started successfully! (GPU: " << prop.name
              << " | VRAM: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)\n";

    // 2. SEED CONFIGURATION (RANDOMLY GENERATED AND DISPLAYED)
    std::random_device rd;
    const uint64_t SAMPLE_SEED     = (static_cast<uint64_t>(rd()) << 32) | rd();
    const uint64_t SIMULATION_SEED = (static_cast<uint64_t>(rd()) << 32) | rd();

    std::cout << "\n[SEED INFO] Assigned SAMPLE_SEED:     " << SAMPLE_SEED << "\n";
    std::cout << "[SEED INFO] Assigned SIMULATION_SEED: " << SIMULATION_SEED << "\n\n";

    // 3. SIMULATION PARAMETERS
    std::mt19937_64 rng_sample(SAMPLE_SEED);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    std::vector<double> values(SAMPLE_SIZE);
    double denominator = 0.0;
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        values[i] = norm_dist(rng_sample);
        denominator += values[i] * values[i];
    }
    const double inv_denominator = 1.0 / denominator;

    // Copy base sample to GPU constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_values, values.data(), SAMPLE_SIZE * sizeof(double)));

    const std::vector<long long> target_simulations = {
        10LL, 100LL, 1000LL, 10000LL, 100000LL,
        1000000LL, 10000000LL, 100000000LL, 1000000000LL
    };
    const std::vector<std::string> labels = {
        "10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B"
    };
    const std::vector<std::string> col_headers = {
        "10", "100", "1,000", "10,000", "100,000", "1 M", "10 M", "100 M", "1 B"
    };

    // Allocate a reusable 50M-element GPU buffer (~400 MB VRAM) so even 8GB GPUs run 1B effortlessly
    const long long MAX_BATCH = 50000000LL;
    double* d_batch_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_buffer, MAX_BATCH * sizeof(double)));

    const int threads_per_block = 256;
    std::vector<SimStats> results;
    results.reserve(target_simulations.size());

    // 4. RUN SIMULATIONS (PHASE 1/2)
    print_global_progress("Simulation Run (Phase 1/2)", 10, YELLOW);
    std::cout << "Starting massive GPU simulation up to 1 Billion...\n\n";

    for (size_t idx = 0; idx < target_simulations.size(); ++idx) {
        long long n = target_simulations[idx];
        auto cell_start = std::chrono::high_resolution_clock::now();

        int global_pct = 10 + static_cast<int>((static_cast<double>(idx) / target_simulations.size()) * 60.0);
        print_global_progress("Subset n = " + col_headers[idx], global_pct, YELLOW);

        // Host vector to store all n results (64 GB RAM easily holds 1B doubles = 8 GB)
        std::vector<double> h_dw_values(n);

        long long num_batches = (n + MAX_BATCH - 1) / MAX_BATCH;
        long long step_print = std::max(1LL, num_batches / 20);

        if (num_batches > 1) {
            std::cout << "Processing n = " << col_headers[idx] << " in " << num_batches << " GPU batches:\n";
        }

        for (long long b = 0; b < num_batches; ++b) {
            long long offset = b * MAX_BATCH;
            long long current_count = std::min(MAX_BATCH, n - offset);
            int blocks = static_cast<int>((current_count + threads_per_block - 1) / threads_per_block);

            durbin_watson_kernel<<<blocks, threads_per_block>>>(
                d_batch_buffer,
                current_count,
                SIMULATION_SEED,
                static_cast<uint64_t>(offset),
                inv_denominator
            );
            CUDA_CHECK(cudaGetLastError());

            // Copy batch results from GPU VRAM to Host RAM
            CUDA_CHECK(cudaMemcpy(
                h_dw_values.data() + offset,
                d_batch_buffer,
                current_count * sizeof(double),
                cudaMemcpyDeviceToHost
            ));

            if (num_batches > 1 && ((b + 1) % step_print == 0 || (b + 1) == num_batches)) {
                double progress = static_cast<double>(b + 1) / num_batches;
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - cell_start).count();
                double eta = (progress < 1.0) ? (elapsed / progress) - elapsed : 0.0;

                int filled = static_cast<int>(std::round(20 * progress));
                std::string pbar(filled, '=');
                pbar += std::string(20 - filled, '-');

                std::cout << "\r[" << pbar << "] " << std::setw(3) << static_cast<int>(progress * 100)
                          << "% | Batch " << (b + 1) << "/" << num_batches
                          << " | ETA: " << static_cast<int>(eta / 60) << "m "
                          << static_cast<int>(eta) % 60 << "s   " << std::flush;
            }
        }
        if (num_batches > 1) std::cout << "\n";

        // Compute Summary Statistics on Host
        double sum = 0.0;
        double min_val = h_dw_values[0];
        double max_val = h_dw_values[0];
        for (long long i = 0; i < n; ++i) {
            double v = h_dw_values[i];
            sum += v;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        double mean_val = sum / static_cast<double>(n);

        double sq_diff_sum = 0.0;
        for (long long i = 0; i < n; ++i) {
            double diff = h_dw_values[i] - mean_val;
            sq_diff_sum += diff * diff;
        }
        double sd_val = std::sqrt(sq_diff_sum / static_cast<double>(n - 1));

        double q1_val  = compute_quantile(h_dw_values, 0.25);
        double med_val = compute_quantile(h_dw_values, 0.50);
        double q3_val  = compute_quantile(h_dw_values, 0.75);

        auto cell_end = std::chrono::high_resolution_clock::now();
        double runtime_sec = std::chrono::duration<double>(cell_end - cell_start).count();

        results.push_back({
            labels[idx], n, runtime_sec, mean_val, med_val, min_val, max_val, q1_val, q3_val, sd_val
        });

        std::cout << "✓ Completed n = " << col_headers[idx] << " in "
                  << std::fixed << std::setprecision(2) << runtime_sec << " s\n\n";
    }

    CUDA_CHECK(cudaFree(d_batch_buffer));

    // 5. PRINT SUMMARY TABLE & EXPORT CSV
    std::cout << "\n" << std::left << std::setw(14) << "Index";
    for (const auto& h : col_headers) std::cout << std::right << std::setw(12) << h;
    std::cout << "\n" << std::string(14 + 12 * col_headers.size(), '-') << "\n";

    auto print_row = [&](const std::string& row_name, auto getter, int prec) {
        std::cout << std::left << std::setw(14) << row_name;
        for (const auto& r : results) {
            std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(prec) << getter(r);
        }
        std::cout << "\n";
    };

    print_row("Runtime (s)", [](const SimStats& s) { return s.runtime_sec; }, 2);
    print_row("Mean",        [](const SimStats& s) { return s.mean; }, 5);
    print_row("Median",      [](const SimStats& s) { return s.median; }, 5);
    print_row("Min",         [](const SimStats& s) { return s.min_val; }, 5);
    print_row("Max",         [](const SimStats& s) { return s.max_val; }, 5);
    print_row("Q1",          [](const SimStats& s) { return s.q1; }, 5);
    print_row("Q3",          [](const SimStats& s) { return s.q3; }, 5);
    print_row("SD",          [](const SimStats& s) { return s.sd; }, 5);

    const std::string csv_filename = "durbin_watson_simulation_results_cuda.csv";
    std::ofstream csv_file(csv_filename);
    csv_file << "Index";
    for (const auto& h : col_headers) csv_file << ",\"" << h << "\"";
    csv_file << "\n";

    auto write_csv_row = [&](const std::string& row_name, auto getter, int prec) {
        csv_file << row_name;
        for (const auto& r : results) {
            csv_file << "," << std::fixed << std::setprecision(prec) << getter(r);
        }
        csv_file << "\n";
    };

    write_csv_row("Runtime (s)", [](const SimStats& s) { return s.runtime_sec; }, 2);
    write_csv_row("Mean",        [](const SimStats& s) { return s.mean; }, 5);
    write_csv_row("Median",      [](const SimStats& s) { return s.median; }, 5);
    write_csv_row("Min",         [](const SimStats& s) { return s.min_val; }, 5);
    write_csv_row("Max",         [](const SimStats& s) { return s.max_val; }, 5);
    write_csv_row("Q1",          [](const SimStats& s) { return s.q1; }, 5);
    write_csv_row("Q3",          [](const SimStats& s) { return s.q3; }, 5);
    write_csv_row("SD",          [](const SimStats& s) { return s.sd; }, 5);
    csv_file.close();

    std::cout << "\n[SUCCESS] Matrix data exported to '" << csv_filename << "'\n";
    print_global_progress("Simulation Completed", 75, YELLOW);

    // 6. BOXPLOT DATA & GNUPLOT SCRIPT GENERATION (PHASE 2/2)
    print_global_progress("Generating Boxplots (Phase 2/2)", 80, GREEN);

    const std::string dat_filename = "dw_boxplot_stats_cuda.dat";
    std::ofstream dat_file(dat_filename);
    dat_file << "# idx label whisker_low q1 median q3 whisker_high mean\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        double iqr = r.q3 - r.q1;
        double w_low  = std::max(r.min_val, r.q1 - 1.5 * iqr);
        double w_high = std::min(r.max_val, r.q3 + 1.5 * iqr);
        dat_file << (i + 1) << " \"" << r.label << "\" "
                 << w_low << " " << r.q1 << " " << r.median << " "
                 << r.q3 << " " << w_high << " " << r.mean << "\n";
    }
    dat_file.close();

    const std::string gp_filename = "plot_dw_cuda.gp";
    std::ofstream gp_file(gp_filename);
    gp_file << "set terminal pngcairo size 2000,1000 enhanced font 'Arial,14'\n"
            << "set output 'dw_simulation_boxplots_cuda.png'\n"
            << "set title 'Durbin-Watson Statistic Distribution with Mean and Median (Up to 1 Billion)' font 'Arial,18'\n"
            << "set xlabel 'Subset Size (n)' font 'Arial,15'\n"
            << "set ylabel 'd-Statistic Value' font 'Arial,15'\n"
            << "set yrange [0:4]\n"
            << "set xrange [0.3:" << (results.size() + 0.7) << "]\n"
            << "set grid ytics lt 0 lw 1 lc rgb '#888888'\n"
            << "set key top right box opaque\n"
            << "set boxwidth 0.45\n"
            << "set style fill solid 0.85 border lc rgb '#2c3e50'\n"
            << "set arrow from 0.3,2.0 to " << (results.size() + 0.7) << ",2.0 nohead dt 2 lw 2 lc rgb 'red'\n"
            << "plot '" << dat_filename << "' using 1:4:3:7:6:xtic(2) with candlesticks lc rgb '#2c3e50' lw 1.5 fc rgb '#a6c8e0' title 'IQR & 1.5x Whiskers' whiskerbars 0.5, \\\n"
            << "     '' using 1:5:5:5:5 with candlesticks lc rgb '#000000' lw 2.5 title 'Sample Median', \\\n"
            << "     '' using 1:8 with points pt 9 ps 1.8 lc rgb '#2ecc71' title 'Sample Mean', \\\n"
            << "     NaN with lines dt 2 lw 2 lc rgb 'red' title 'No Autocorrelation (d = 2)'\n";
    gp_file.close();

    int gp_status = std::system("gnuplot plot_dw_cuda.gp 2>/dev/null");
    if (gp_status == 0) {
        std::cout << "[SUCCESS] Boxplot saved to 'dw_simulation_boxplots_cuda.png'\n";
    } else {
        std::cout << "[INFO] Gnuplot script saved to '" << gp_filename << "'. Install gnuplot and run 'gnuplot " << gp_filename << "' to render the PNG.\n";
    }

    print_global_progress("Execution Finished", 100, GREEN);

    // 7. TOTAL RUNTIME
    auto global_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(global_end - global_start).count();
    std::cout << "\n[TIMER] Total execution time: " << std::fixed << std::setprecision(2)
              << total_elapsed << " seconds (" << static_cast<int>(total_elapsed / 60) << "m "
              << static_cast<int>(total_elapsed) % 60 << "s)\n";

    return 0;
}