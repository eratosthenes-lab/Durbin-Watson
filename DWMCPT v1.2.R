if (!requireNamespace("Rfast", quietly = TRUE)) {
  install.packages("Rfast")
}

# ==============================================================================
# DURBIN-WATSON MASSIVE SIMULATION - OPTIMIZED FOR RSTUDIO (64 GB RAM)
# ==============================================================================

start_time <- Sys.time()

# 1. PACKAGE CHECK
use_rfast <- requireNamespace("Rfast", quietly = TRUE)
if (use_rfast) {
  cat("[INFO] 'Rfast' package detected: C++ acceleration enabled.\n")
} else {
  cat("[WARNING] 'Rfast' package not installed. To speed up execution by ~20x, run:\n")
  cat("          install.packages('Rfast')\n\n")
}

# 2. PROGRESS BAR FUNCTION OPTIMIZED FOR RSTUDIO CONSOLE
print_global_progress <- function(phase_name, progress_percentage, color = "cyan") {
  colors <- list(
    cyan   = "\033[96m",
    yellow = "\033[93m",
    green  = "\033[92m",
    reset  = "\033[0m"
  )
  c_code <- colors[[color]]
  reset  <- colors[["reset"]]
  
  bar_length <- 30
  filled_length <- round(bar_length * (progress_percentage / 100))
  bar <- paste0(
    paste(rep("█", filled_length), collapse = ""),
    paste(rep("-", bar_length - filled_length), collapse = "")
  )
  
  cat(sprintf("%s[GLOBAL PROGRESS] [%s] %3d%% | %s%s\n", 
              c_code, bar, progress_percentage, phase_name, reset))
  flush.console()
}

print_global_progress("Initialization", 0, "cyan")

# 3. SEED CONFIGURATION (RANDOMLY GENERATED AND DISPLAYED)
# Set two fixed integers manually to reproduce a specific run
SAMPLE_SEED     <- sample.int(.Machine$integer.max, 1)
SIMULATION_SEED <- sample.int(.Machine$integer.max, 1)

cat(sprintf("\n[SEED INFO] Assigned SAMPLE_SEED:     %d\n", SAMPLE_SEED))
cat(sprintf("[SEED INFO] Assigned SIMULATION_SEED: %d\n\n", SIMULATION_SEED))

# 4. SIMULATION PARAMETERS
sample_size <- 50
set.seed(SAMPLE_SEED)
values <- rnorm(sample_size)
denominator <- sum(values^2)

target_simulations <- c(10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
labels <- c("10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B")

metrics <- list(
  "Runtime (s)" = character(),
  "Mean"        = character(),
  "Median"      = character(),
  "Min"         = character(),
  "Max"         = character(),
  "Q1"          = character(),
  "Q3"          = character(),
  "SD"          = character()
)

dw_simulations_history <- vector("list", length(target_simulations))
names(dw_simulations_history) <- labels

# High-performance permutation and Durbin-Watson computation function
compute_dw_chunk <- function(chunk_rows, vals, denom, fast_mode) {
  rand_mat <- matrix(runif(chunk_rows * length(vals)), nrow = chunk_rows, ncol = length(vals))
  
  if (fast_mode) {
    indices <- Rfast::rowOrder(rand_mat)
  } else {
    indices <- t(apply(rand_mat, 1, order))
  }
  
  perm_mat <- matrix(vals[indices], nrow = chunk_rows, ncol = length(vals))
  diffs <- perm_mat[, -1, drop = FALSE] - perm_mat[, -ncol(perm_mat), drop = FALSE]
  return(rowSums(diffs^2) / denom)
}

# 5. RUN SIMULATIONS (PHASE 1/2)
print_global_progress("Simulation Run (Phase 1/2)", 10, "yellow")

for (idx in seq_along(target_simulations)) {
  n <- target_simulations[idx]
  cell_start_time <- Sys.time()
  
  # Re-initialize the RNG with the exact same SIMULATION_SEED for every simulation
  set.seed(SIMULATION_SEED)
  
  global_pct <- 10 + as.integer(((idx - 1) / length(target_simulations)) * 60)
  print_global_progress(sprintf("Subset n = %s", format(n, big.mark = ",", scientific = FALSE)), global_pct, "yellow")
  
  if (n <= 1e7) {
    dw_values <- compute_dw_chunk(n, values, denominator, use_rfast)
  } else {
    batch_size  <- 1e7
    num_batches <- n / batch_size
    dw_values   <- numeric(n)
    step_print  <- max(1, floor(num_batches / 20)) # Print every 5% (every 5 batches out of 100)
    
    for (i in seq_len(num_batches)) {
      start_idx <- (i - 1) * batch_size + 1
      end_idx   <- i * batch_size
      dw_values[start_idx:end_idx] <- compute_dw_chunk(batch_size, values, denominator, use_rfast)
      
      if (i %% 5 == 0) gc(verbose = FALSE)
      
      if (i %% step_print == 0 || i == num_batches) {
        progress <- i / num_batches
        elapsed  <- as.numeric(difftime(Sys.time(), cell_start_time, units = "secs"))
        eta      <- ifelse(progress < 1.0, (elapsed / progress) - elapsed, 0)
        eta_str  <- ifelse(eta > 0, sprintf("%dm %ds", as.integer(eta %/% 60), as.integer(eta %% 60)), "0s")
        
        cat(sprintf("   -> Batch %3d/%d (%3d%%) | ETA: %s\n", i, num_batches, as.integer(progress * 100), eta_str))
        flush.console()
      }
    }
  }
  
  dw_simulations_history[[labels[idx]]] <- dw_values
  
  runtime_sec <- as.numeric(difftime(Sys.time(), cell_start_time, units = "secs"))
  quants <- quantile(dw_values, probs = c(0.25, 0.50, 0.75), names = FALSE)
  
  metrics[["Runtime (s)"]] <- c(metrics[["Runtime (s)"]], sprintf("%.1f", runtime_sec))
  metrics[["Mean"]]        <- c(metrics[["Mean"]],        sprintf("%.5f", mean(dw_values)))
  metrics[["Median"]]      <- c(metrics[["Median"]],      sprintf("%.5f", quants[2]))
  metrics[["Min"]]         <- c(metrics[["Min"]],         sprintf("%.5f", min(dw_values)))
  metrics[["Max"]]         <- c(metrics[["Max"]],         sprintf("%.5f", max(dw_values)))
  metrics[["Q1"]]          <- c(metrics[["Q1"]],          sprintf("%.5f", quants[1]))
  metrics[["Q3"]]          <- c(metrics[["Q3"]],          sprintf("%.5f", quants[3]))
  metrics[["SD"]]          <- c(metrics[["SD"]],          sprintf("%.5f", sd(dw_values)))
  
  cat(sprintf("✓ Completed n = %s in %.1f s\n\n", format(n, big.mark = ",", scientific = FALSE), runtime_sec))
  gc(verbose = FALSE)
}

# 6. SUMMARY TABLE AND CSV EXPORT
formatted_headers <- c("Index", "10", "100", "1,000", "10,000", "100,000", "1 M", "10 M", "100 M", "1 B")
formatted_headers <- formatted_headers[seq_len(length(target_simulations) + 1)]

table_df <- do.call(rbind, metrics)
table_df <- cbind(Index = rownames(table_df), as.data.frame(table_df, stringsAsFactors = FALSE))
rownames(table_df) <- NULL
colnames(table_df) <- formatted_headers

if (requireNamespace("knitr", quietly = TRUE)) {
  print(knitr::kable(table_df, format = "pipe"))
} else {
  print(table_df, row.names = FALSE)
}

output_filename <- "durbin_watson_simulation_results_R.csv"
write.csv(table_df, file = output_filename, row.names = FALSE)
cat(sprintf("\n[SUCCESS] Data exported to: %s/%s\n", getwd(), output_filename))

print_global_progress("Simulation Completed", 75, "yellow")

# 7. BOXPLOT GENERATION (FILE EXPORT + RSTUDIO PREVIEW)
print_global_progress("Generating Boxplots (Phase 2/2)", 80, "green")

render_dw_boxplot <- function(data_list, x_labels) {
  par(mar = c(5.5, 5.5, 4.5, 2) + 0.1)
  
  boxplot(
    data_list,
    names     = x_labels,
    col       = "#a6c8e0",
    border    = "#2c3e50",
    medcol    = "#000000",
    medlwd    = 2.5,
    whisklty  = 1,
    whisklwd  = 1.5,
    whiskcol  = "#2c3e50",
    staplecol = "#2c3e50",
    ylim      = c(0, 4),
    main      = "Durbin-Watson Statistic Distribution with Mean and Median",
    xlab      = "Subset Size (n)",
    ylab      = "d-Statistic Value",
    cex.main  = 1.6,
    cex.lab   = 1.3,
    cex.axis  = 1.1,
    outline   = FALSE
  )
  
  grid(nx = NA, ny = NULL, lty = 3, col = adjustcolor("gray50", alpha.f = 0.6))
  abline(h = 2.0, col = "red", lty = 2, lwd = 1.5)
  
  means <- sapply(data_list, mean)
  points(seq_along(means), means, pch = 17, col = "#2ecc71", cex = 1.4)
  
  legend(
    "topright",
    legend = c("No Autocorrelation (d = 2)", "Sample Mean", "Sample Median"),
    col    = c("red", "#2ecc71", "#000000"),
    lty    = c(2, NA, 1),
    lwd    = c(1.5, NA, 2.5),
    pch    = c(NA, 17, NA),
    pt.cex = 1.4,
    cex    = 1.0,
    bg     = "white"
  )
}

# A) Save high-resolution PNG file
output_image_name <- "dw_simulation_boxplots_R.png"
png(filename = output_image_name, width = 20, height = 10, units = "in", res = 400)
render_dw_boxplot(dw_simulations_history, labels)
dev.off()

# B) Render directly in RStudio Plots pane
render_dw_boxplot(dw_simulations_history, labels)

print_global_progress("Execution Finished", 100, "green")

# 8. TOTAL RUNTIME AND DATA VIEWER IN RSTUDIO
total_elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
cat(sprintf("\n[TIMER] Total execution time: %.2f seconds (%dm %ds)\n", 
            total_elapsed, as.integer(total_elapsed %/% 60), as.integer(total_elapsed %% 60)))

if (interactive()) {
  View(table_df, title = "Durbin-Watson Results")
}