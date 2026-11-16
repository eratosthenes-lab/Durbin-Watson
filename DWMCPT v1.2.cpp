// ==============================================================================
// DURBIN-WATSON MASSIVE SIMULATION - HIGH-PERFORMANCE C++17 VERSION (OPENMP)
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
#include <numeric>
#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

// Struct to store summary statistics for each subset size
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

// Inline Durbin-Watson statistic computation on a permuted vector
inline double compute_dw_single(const std::vector<double>& perm_vals, double inv_denom) {
    double num = 0.0;
    const size_t sz = perm_vals.size();
    for (size_t j = 0; j < sz - 1; ++j) {
        double d = perm_vals[j + 1] - perm_vals[j];
        num += d * d;
    }
    return num * inv_denom;
}

// Fast quantile calculation using std::nth_element (O(N) average, low memory overhead)
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

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    print_global_progress("Initialization", 0, CYAN);
    std::cout << "Global timer started successfully! (Active CPU threads: " << num_threads << ")\n";

    // 2. SEED CONFIGURATION (RANDOMLY GENERATED AND DISPLAYED)
    // Replace these two lines with fixed integers to reproduce a specific run
    std::random_device rd;
    const uint64_t SAMPLE_SEED     = (static_cast<uint64_t>(rd()) << 32) | rd();
    const uint64_t SIMULATION_SEED = (static_cast<uint64_t>(rd()) << 32) | rd();

    std::cout << "\n[SEED INFO] Assigned SAMPLE_SEED:     " << SAMPLE_SEED << "\n";
    std::cout << "[SEED INFO] Assigned SIMULATION_SEED: " << SIMULATION_SEED << "\n\n";

    // 3. SIMULATION PARAMETERS
    const size_t SAMPLE_SIZE = 50;
    std::mt19937_64 rng_sample(SAMPLE_SEED);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    std::vector<double> values(SAMPLE_SIZE);
    double denominator = 0.0;
    for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
        values[i] = norm_dist(rng_sample);
        denominator += values[i] * values[i];
    }
    const double inv_denominator = 1.0 / denominator;

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

    std::vector<SimStats> results;
    results.reserve(target_simulations.size());

    // 4. RUN SIMULATIONS (PHASE 1/2)
    print_global_progress("Simulation Run (Phase 1/2)", 10, YELLOW);
    std::cout << "Starting massive simulation up to 1 Billion...\n\n";

    for (size_t idx = 0; idx < target_simulations.size(); ++idx) {
        long long n = target_simulations[idx];
        auto cell_start = std::chrono::high_resolution_clock::now();

        int global_pct = 10 + static_cast<int>((static_cast<double>(idx) / target_simulations.size()) * 60.0);
        print_global_progress("Subset n = " + col_headers[idx], global_pct, YELLOW);

        std::vector<double> dw_values(n);

        if (n <= 10000000LL) {
            // Deterministic single-thread execution for small/medium n
            std::mt19937_64 sim_rng(SIMULATION_SEED);
            std::vector<double> buf = values;
            for (long long i = 0; i < n; ++i) {
                std::shuffle(buf.begin(), buf.end(), sim_rng);
                dw_values[i] = compute_dw_single(buf, inv_denominator);
            }
        } else {
            // Batched & Multi-threaded OpenMP execution for 100M and 1B
            const long long batch_size = 10000000LL;
            const long long num_batches = n / batch_size;
            const long long step_print = std::max(1LL, num_batches / 20);

            std::cout << "Processing n = " << col_headers[idx] << " in " << num_batches
                      << " batches across " << num_threads << " threads:\n";

            // Derive deterministic sub-seeds from SIMULATION_SEED
            std::mt19937_64 master_rng(SIMULATION_SEED);
            std::vector<uint64_t> batch_seeds(num_batches);
            for (long long b = 0; b < num_batches; ++b) {
                batch_seeds[b] = master_rng();
            }

            for (long long b = 0; b < num_batches; ++b) {
                long long start_idx = b * batch_size;
                long long end_idx = start_idx + batch_size;

                #pragma omp parallel
                {
                    int tid = 0;
                    int nthreads = 1;
#ifdef _OPENMP
                    tid = omp_get_thread_num();
                    nthreads = omp_get_num_threads();
#endif
                    // Split batch evenly across threads with thread-local RNG & buffer
                    long long chunk = (batch_size + nthreads - 1) / nthreads;
                    long long t_start = start_idx + tid * chunk;
                    long long t_end = std::min(end_idx, t_start + chunk);

                    if (t_start < t_end) {
                        std::mt19937_64 local_rng(batch_seeds[b] + static_cast<uint64_t>(tid) * 0x9e3779b97f4a7c15ULL);
                        std::vector<double> local_buf = values;
                        for (long long i = t_start; i < t_end; ++i) {
                            std::shuffle(local_buf.begin(), local_buf.end(), local_rng);
                            dw_values[i] = compute_dw_single(local_buf, inv_denominator);
                        }
                    }
                }

                if ((b + 1) % step_print == 0 || (b + 1) == num_batches) {
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
            std::cout << "\n";
        }

        // Compute Mean & Standard Deviation in parallel
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (long long i = 0; i < n; ++i) {
            sum += dw_values[i];
        }
        double mean_val = sum / static_cast<double>(n);

        double sq_diff_sum = 0.0;
        #pragma omp parallel for reduction(+:sq_diff_sum)
        for (long long i = 0; i < n; ++i) {
            double diff = dw_values[i] - mean_val;
            sq_diff_sum += diff * diff;
        }
        double sd_val = std::sqrt(sq_diff_sum / static_cast<double>(n - 1));

        // Compute Min, Max, Q1, Median, Q3 in-place
        auto minmax_it = std::minmax_element(dw_values.begin(), dw_values.end());
        double min_val = *minmax_it.first;
        double max_val = *minmax_it.second;
        double q1_val  = compute_quantile(dw_values, 0.25);
        double med_val = compute_quantile(dw_values, 0.50);
        double q3_val  = compute_quantile(dw_values, 0.75);

        auto cell_end = std::chrono::high_resolution_clock::now();
        double runtime_sec = std::chrono::duration<double>(cell_end - cell_start).count();

        results.push_back({
            labels[idx], n, runtime_sec, mean_val, med_val, min_val, max_val, q1_val, q3_val, sd_val
        });

        std::cout << "✓ Completed n = " << col_headers[idx] << " in "
                  << std::fixed << std::setprecision(2) << runtime_sec << " s\n\n";
    }

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

    // Export CSV
    const std::string csv_filename = "durbin_watson_simulation_results_cpp.csv";
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

    // Compute Tukey boxplot whiskers (1.5 * IQR) and write stats for Gnuplot
    const std::string dat_filename = "dw_boxplot_stats_cpp.dat";
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

    const std::string gp_filename = "plot_dw_cpp.gp";
    std::ofstream gp_file(gp_filename);
    gp_file << "set terminal pngcairo size 2000,1000 enhanced font 'Arial,14'\n"
            << "set output 'dw_simulation_boxplots_cpp.png'\n"
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

    // Automatically invoke gnuplot if installed in system PATH
    int gp_status = std::system("gnuplot plot_dw_cpp.gp 2>/dev/null");
    if (gp_status == 0) {
        std::cout << "[SUCCESS] Boxplot saved to 'dw_simulation_boxplots_cpp.png'\n";
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