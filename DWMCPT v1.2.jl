# ==============================================================================
# DURBIN-WATSON MASSIVE SIMULATION - HIGH-PERFORMANCE JULIA VERSION
# ==============================================================================

# 0. AUTOMATIC PACKAGE CHECK & INSTALLATION (RUNS ONLY IF PACKAGES ARE MISSING)
using Pkg
const REQUIRED_PACKAGES = ["PrettyTables", "Plots", "StatsPlots", "CSV", "DataFrames"]

for pkg in REQUIRED_PACKAGES
    if Base.find_package(pkg) === nothing
        println("[SETUP] Installing missing package: $pkg ...")
        Pkg.add(pkg)
    end
end

using Random
using Statistics
using Printf
using PrettyTables
using DataFrames
using CSV
using Plots
using StatsPlots

start_time = time()

# 1. GLOBAL PROGRESS BAR FUNCTION
function print_global_progress(phase_name::String, progress_percentage::Int, color::Symbol = :cyan)
    bar_length = 30
    filled_length = round(Int, bar_length * (progress_percentage / 100))
    bar = repeat("█", filled_length) * repeat("-", bar_length - filled_length)
    printstyled("[GLOBAL PROGRESS] [$bar] $(lpad(progress_percentage, 3))% | Phase: $phase_name\n", color = color)
    flush(stdout)
end

print_global_progress("Initialization", 0, :cyan)
println("Global timer started successfully! (Active CPU threads: $(Threads.nthreads()))")

# 2. SEED CONFIGURATION (RANDOMLY GENERATED AND DISPLAYED)
# Set two fixed integers manually to reproduce a specific run
const SAMPLE_SEED     = rand(1:4_000_000_000)
const SIMULATION_SEED = rand(1:4_000_000_000)

println("\n[SEED INFO] Assigned SAMPLE_SEED:     $SAMPLE_SEED")
println("[SEED INFO] Assigned SIMULATION_SEED: $SIMULATION_SEED\n")

# 3. SIMULATION PARAMETERS
const SAMPLE_SIZE = 50
rng_sample = Xoshiro(SAMPLE_SEED)
const VALUES = randn(rng_sample, SAMPLE_SIZE)
const DENOMINATOR = sum(abs2, VALUES)

const TARGET_SIMULATIONS = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
const LABELS = ["10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B"]

# Zero-allocation Durbin-Watson calculation on a permuted vector
@inline function compute_dw_single(perm_vals::Vector{Float64}, denom::Float64)::Float64
    num = 0.0
    @inbounds @simd for j in 1:(length(perm_vals) - 1)
        d = perm_vals[j + 1] - perm_vals[j]
        num += d * d
    end
    return num / denom
end

# 4. RUN SIMULATIONS (PHASE 1/2)
print_global_progress("Simulation Run (Phase 1/2)", 10, :yellow)
println("Starting massive simulation up to 1 Billion...\n")

metrics = Dict{String, Vector{String}}(
    "Runtime (s)" => String[],
    "Mean"        => String[],
    "Median"      => String[],
    "Min"         => String[],
    "Max"         => String[],
    "Q1"          => String[],
    "Q3"          => String[],
    "SD"          => String[]
)

dw_simulations_history = Vector{Vector{Float64}}(undef, length(TARGET_SIMULATIONS))

for (idx, n) in enumerate(TARGET_SIMULATIONS)
    cell_start_time = time()
    
    global_pct = 10 + floor(Int, ((idx - 1) / length(TARGET_SIMULATIONS)) * 60)
    print_global_progress("Subset n = $(n)", global_pct, :yellow)
    
    dw_values = Vector{Float64}(undef, n)
    
    if n <= 10_000_000
        # Re-initialize RNG with exact same SIMULATION_SEED for deterministic reproducibility
        sim_rng = Xoshiro(SIMULATION_SEED)
        buf = copy(VALUES)
        for i in 1:n
            shuffle!(sim_rng, buf)
            @inbounds dw_values[i] = compute_dw_single(buf, DENOMINATOR)
        end
    else
        # Batched & Multi-threaded execution for 100M and 1B
        batch_size  = 10_000_000
        num_batches = n ÷ batch_size
        step_print  = max(1, num_batches ÷ 20)
        
        println("Processing n = $n in $num_batches batches across $(Threads.nthreads()) threads:")
        
        # Derive deterministic sub-seeds from SIMULATION_SEED for each batch
        master_rng = Xoshiro(SIMULATION_SEED)
        batch_seeds = rand(master_rng, UInt64, num_batches)
        
        for b in 1:num_batches
            start_idx = (b - 1) * batch_size + 1
            end_idx   = b * batch_size
            
            # Parallel execution inside the batch with thread-local buffers
            Threads.@threads for tid in 1:Threads.nthreads()
                chunk_len = cld(batch_size, Threads.nthreads())
                t_start = start_idx + (tid - 1) * chunk_len
                t_end   = min(end_idx, t_start + chunk_len - 1)
                
                if t_start <= t_end
                    local_rng = Xoshiro(batch_seeds[b] + tid)
                    local_buf = copy(VALUES)
                    for i in t_start:t_end
                        shuffle!(local_rng, local_buf)
                        @inbounds dw_values[i] = compute_dw_single(local_buf, DENOMINATOR)
                    end
                end
            end
            
            if b % step_print == 0 || b == num_batches
                progress = b / num_batches
                elapsed  = time() - cell_start_time
                eta      = progress < 1.0 ? (elapsed / progress) - elapsed : 0.0
                eta_str  = eta > 0 ? "$(floor(Int, eta ÷ 60))m $(floor(Int, eta % 60))s" : "0s"
                
                filled = round(Int, 20 * progress)
                bar = repeat("=", filled) * repeat("-", 20 - filled)
                print("\r[$bar] $(round(Int, progress * 100))% | Batch $b/$num_batches | ETA: $eta_str   ")
                flush(stdout)
            end
        end
        println()
    end
    
    dw_simulations_history[idx] = dw_values
    
    runtime_sec = time() - cell_start_time
    q1, med, q3 = quantile(dw_values, [0.25, 0.50, 0.75])
    
    push!(metrics["Runtime (s)"], @sprintf("%.2f", runtime_sec))
    push!(metrics["Mean"],        @sprintf("%.5f", mean(dw_values)))
    push!(metrics["Median"],      @sprintf("%.5f", med))
    push!(metrics["Min"],         @sprintf("%.5f", minimum(dw_values)))
    push!(metrics["Max"],         @sprintf("%.5f", maximum(dw_values)))
    push!(metrics["Q1"],          @sprintf("%.5f", q1))
    push!(metrics["Q3"],          @sprintf("%.5f", q3))
    push!(metrics["SD"],          @sprintf("%.5f", std(dw_values)))
    
    println("✓ Completed n = $n in $(@sprintf("%.2f", runtime_sec)) s\n")
end

# 5. SUMMARY TABLE AND CSV EXPORT
formatted_headers = ["Index", "10", "100", "1,000", "10,000", "100,000", "1 M", "10 M", "100 M", "1 B"]
metric_order = ["Runtime (s)", "Mean", "Median", "Min", "Max", "Q1", "Q3", "SD"]

table_matrix = Matrix{String}(undef, length(metric_order), length(formatted_headers))
for (r, m_name) in enumerate(metric_order)
    table_matrix[r, 1] = m_name
    table_matrix[r, 2:end] = metrics[m_name]
end

pretty_table(table_matrix; header = formatted_headers)

output_filename = "durbin_watson_simulation_results_julia.csv"
df_export = DataFrame(table_matrix, formatted_headers)
CSV.write(output_filename, df_export)
println("\n[SUCCESS] Matrix data exported to '$output_filename'")

print_global_progress("Simulation Completed", 75, :yellow)

# 6. BOXPLOT GENERATION (PHASE 2/2)
print_global_progress("Generating Boxplots (Phase 2/2)", 80, :green)

means = [mean(v) for v in dw_simulations_history]

p = boxplot(
    permutedims(LABELS),
    dw_simulations_history,
    fillalpha = 0.85,
    fillcolor = "#a6c8e0",
    linecolor = "#2c3e50",
    linewidth = 1.5,
    outliers = false,
    legend = :topright,
    label = "",
    size = (1600, 800),
    dpi = 400,
    ylim = (0, 4),
    title = "Durbin-Watson Statistic Distribution with Mean and Median (Up to 1 Billion)",
    xlabel = "Subset Size (n)",
    ylabel = "d-Statistic Value",
    titlefontsize = 16,
    guidefontsize = 14,
    tickfontsize = 12,
    grid = (:y, :dot, 0.6)
)

# Reference line d = 2
hline!(p, [2.0], color = :red, linestyle = :dash, linewidth = 1.5, label = "No Autocorrelation (d = 2)")

# Overlay sample means (green triangles)
scatter!(p, 1:length(LABELS), means, marker = (:utriangle, 8, "#2ecc71"), markerstrokecolor = "#27ae60", label = "Sample Mean")

# Dummy entry for median legend
plot!(p, Float64[], Float64[], color = :black, linewidth = 2.5, label = "Sample Median")

output_image_name = "dw_simulation_boxplots_julia.png"
savefig(p, output_image_name)
display(p)

print_global_progress("Plot Generated and Saved", 95, :green)
print_global_progress("Execution Finished", 100, :green)

# 7. TOTAL RUNTIME
total_elapsed = time() - start_time
println("\n[TIMER] Total execution time: $(@sprintf("%.2f", total_elapsed)) seconds ($(floor(Int, total_elapsed ÷ 60))m $(floor(Int, total_elapsed % 60))s)")