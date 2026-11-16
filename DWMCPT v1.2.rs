---
[package]
edition = "2021"

[dependencies]
rand = "0.8"
rand_xoshiro = "0.6"
rayon = "1.10"
plotters = "0.3"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
target-cpu = "native"
---

// ==============================================================================
// DURBIN-WATSON MASSIVE SIMULATION - SINGLE-FILE RUST VERSION
// ==============================================================================

use plotters::prelude::*;
use rand::distributions::{Distribution, Standard};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

struct SimStats {
    label: String,
    runtime_sec: f64,
    mean: f64,
    median: f64,
    min_val: f64,
    max_val: f64,
    q1: f64,
    q3: f64,
    sd: f64,
}

fn generate_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn print_global_progress(phase_name: &str, progress_percentage: usize, color_code: &str) {
    let bar_length = 30;
    let filled_length = ((bar_length as f64) * (progress_percentage as f64 / 100.0)).round() as usize;
    let bar = format!(
        "{}{}",
        "█".repeat(filled_length),
        "-".repeat(bar_length.saturating_sub(filled_length))
    );

    println!(
        "{}[GLOBAL PROGRESS] [{}] {:>3}% | Phase: {}\x1b[0m",
        color_code, bar, progress_percentage, phase_name
    );
    let _ = io::stdout().flush();
}

#[inline(always)]
fn compute_dw_single(perm_vals: &[f64], inv_denom: f64) -> f64 {
    let mut num = 0.0;
    for j in 0..(perm_vals.len() - 1) {
        let d = perm_vals[j + 1] - perm_vals[j];
        num += d * d;
    }
    num * inv_denom
}

fn compute_quantile(data: &mut [f64], q: f64) -> f64 {
    let n = data.len();
    let pos = q * ((n - 1) as f64);
    let idx = pos.floor() as usize;
    let frac = pos - (idx as f64);

    let (_, val1_ref, right_slice) =
        data.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    let val1 = *val1_ref;

    if frac == 0.0 || right_slice.is_empty() {
        return val1;
    }

    let val2 = right_slice
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    val1 + frac * (val2 - val1)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const CYAN: &str = "\x1b[96m";
    const YELLOW: &str = "\x1b[93m";
    const GREEN: &str = "\x1b[92m";

    let global_start = Instant::now();
    let num_threads = rayon::current_num_threads();

    print_global_progress("Initialization", 0, CYAN);
    println!(
        "Global timer started successfully! (Active CPU threads: {})",
        num_threads
    );

    let mut sys_rng = rand::thread_rng();
    let sample_seed: u64 = sys_rng.gen();
    let simulation_seed: u64 = sys_rng.gen();

    println!("\n[SEED INFO] Assigned SAMPLE_SEED:     {}", sample_seed);
    println!("[SEED INFO] Assigned SIMULATION_SEED: {}\n", simulation_seed);

    const SAMPLE_SIZE: usize = 50;
    let mut rng_sample = Xoshiro256PlusPlus::seed_from_u64(sample_seed);
    let values: Vec<f64> = (0..SAMPLE_SIZE)
        .map(|_| generate_standard_normal(&mut rng_sample))
        .collect();

    let denominator: f64 = values.iter().map(|x| x * x).sum();
    let inv_denominator = 1.0 / denominator;

    let target_simulations: [usize; 9] = [
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
    ];
    let labels = ["10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B"];
    let col_headers = [
        "10", "100", "1,000", "10,000", "100,000", "1 M", "10 M", "100 M", "1 B",
    ];

    let mut results: Vec<SimStats> = Vec::with_capacity(target_simulations.len());

    print_global_progress("Simulation Run (Phase 1/2)", 10, YELLOW);
    println!("Starting massive simulation up to 1 Billion...\n");

    for (idx, &n) in target_simulations.iter().enumerate() {
        let cell_start = Instant::now();

        let global_pct = 10 + ((idx as f64 / target_simulations.len() as f64) * 60.0) as usize;
        print_global_progress(&format!("Subset n = {}", col_headers[idx]), global_pct, YELLOW);

        let mut dw_values: Vec<f64> = vec![0.0; n];

        if n <= 10_000_000 {
            let mut sim_rng = Xoshiro256PlusPlus::seed_from_u64(simulation_seed);
            let mut buf = values.clone();
            for val in dw_values.iter_mut() {
                buf.shuffle(&mut sim_rng);
                *val = compute_dw_single(&buf, inv_denominator);
            }
        } else {
            let batch_size: usize = 10_000_000;
            let num_batches = n / batch_size;
            let step_print = (num_batches / 20).max(1);

            println!(
                "Processing n = {} in {} batches across {} threads:",
                col_headers[idx], num_batches, num_threads
            );

            let mut master_rng = Xoshiro256PlusPlus::seed_from_u64(simulation_seed);
            let batch_seeds: Vec<u64> = Standard.sample_iter(&mut master_rng).take(num_batches).collect();

            for (b, batch_slice) in dw_values.chunks_mut(batch_size).enumerate() {
                let chunk_len = batch_slice.len().div_ceil(num_threads);

                batch_slice
                    .par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(tid, thread_slice)| {
                        let seed = batch_seeds[b]
                            .wrapping_add((tid as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                        let mut local_rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                        let mut local_buf = values.clone();

                        for slot in thread_slice.iter_mut() {
                            local_buf.shuffle(&mut local_rng);
                            *slot = compute_dw_single(&local_buf, inv_denominator);
                        }
                    });

                if (b + 1) % step_print == 0 || (b + 1) == num_batches {
                    let progress = (b + 1) as f64 / num_batches as f64;
                    let elapsed = cell_start.elapsed().as_secs_f64();
                    let eta = if progress < 1.0 {
                        (elapsed / progress) - elapsed
                    } else {
                        0.0
                    };

                    let filled = (20.0 * progress).round() as usize;
                    let pbar = format!("{}{}", "=".repeat(filled), "-".repeat(20 - filled));

                    print!(
                        "\r[{}] {:>3}% | Batch {}/{} | ETA: {}m {}s   ",
                        pbar,
                        (progress * 100.0) as usize,
                        b + 1,
                        num_batches,
                        (eta as u64) / 60,
                        (eta as u64) % 60
                    );
                    let _ = io::stdout().flush();
                }
            }
            println!();
        }

        let sum: f64 = dw_values.par_iter().sum();
        let mean_val = sum / (n as f64);

        let sq_diff_sum: f64 = dw_values
            .par_iter()
            .map(|&x| {
                let d = x - mean_val;
                d * d
            })
            .sum();
        let sd_val = (sq_diff_sum / ((n - 1) as f64)).sqrt();

        let (min_val, max_val) = dw_values
            .par_iter()
            .fold(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(min_a, max_a), &x| (min_a.min(x), max_a.max(x)),
            )
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
            );

        let q1_val = compute_quantile(&mut dw_values, 0.25);
        let med_val = compute_quantile(&mut dw_values, 0.50);
        let q3_val = compute_quantile(&mut dw_values, 0.75);

        let runtime_sec = cell_start.elapsed().as_secs_f64();

        results.push(SimStats {
            label: labels[idx].to_string(),
            runtime_sec,
            mean: mean_val,
            median: med_val,
            min_val,
            max_val,
            q1: q1_val,
            q3: q3_val,
            sd: sd_val,
        });

        println!(
            "✓ Completed n = {} in {:.2} s\n",
            col_headers[idx], runtime_sec
        );
    }

    print!("\n{:<14}", "Index");
    for h in &col_headers {
        print!("{:>12}", h);
    }
    println!("\n{}", "-".repeat(14 + 12 * col_headers.len()));

    let print_row = |name: &str, getter: fn(&SimStats) -> f64, prec: usize| {
        print!("{:<14}", name);
        for r in &results {
            print!("{:>12.*}", prec, getter(r));
        }
        println!();
    };

    print_row("Runtime (s)", |s| s.runtime_sec, 2);
    print_row("Mean", |s| s.mean, 5);
    print_row("Median", |s| s.median, 5);
    print_row("Min", |s| s.min_val, 5);
    print_row("Max", |s| s.max_val, 5);
    print_row("Q1", |s| s.q1, 5);
    print_row("Q3", |s| s.q3, 5);
    print_row("SD", |s| s.sd, 5);

    let csv_filename = "durbin_watson_simulation_results_rust.csv";
    let mut csv_file = File::create(csv_filename)?;
    write!(csv_file, "Index")?;
    for h in &col_headers {
        write!(csv_file, ",\"{}\"", h)?;
    }
    writeln!(csv_file)?;

    let mut write_csv_row = |name: &str, getter: fn(&SimStats) -> f64, prec: usize| -> io::Result<()> {
        write!(csv_file, "{}", name)?;
        for r in &results {
            write!(csv_file, ",{:.*}", prec, getter(r))?;
        }
        writeln!(csv_file)
    };

    write_csv_row("Runtime (s)", |s| s.runtime_sec, 2)?;
    write_csv_row("Mean", |s| s.mean, 5)?;
    write_csv_row("Median", |s| s.median, 5)?;
    write_csv_row("Min", |s| s.min_val, 5)?;
    write_csv_row("Max", |s| s.max_val, 5)?;
    write_csv_row("Q1", |s| s.q1, 5)?;
    write_csv_row("Q3", |s| s.q3, 5)?;
    write_csv_row("SD", |s| s.sd, 5)?;

    println!("\n[SUCCESS] Matrix data exported to '{}'", csv_filename);
    print_global_progress("Simulation Completed", 75, YELLOW);

    print_global_progress("Generating Boxplots (Phase 2/2)", 80, GREEN);

    let output_image_name = "dw_simulation_boxplots_rust.png";
    render_boxplot_png(&results, output_image_name)?;
    println!("[SUCCESS] Boxplot saved to '{}'", output_image_name);

    print_global_progress("Plot Generated and Saved", 95, GREEN);
    print_global_progress("Execution Finished", 100, GREEN);

    let total_elapsed = global_start.elapsed().as_secs_f64();
    println!(
        "\n[TIMER] Total execution time: {:.2} seconds ({}m {}s)",
        total_elapsed,
        (total_elapsed as u64) / 60,
        (total_elapsed as u64) % 60
    );

    Ok(())
}

fn render_boxplot_png(
    results: &[SimStats],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (2000, 1000)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Durbin-Watson Statistic Distribution with Mean and Median (Up to 1 Billion)",
            ("sans-serif", 36).into_font(),
        )
        .margin(30)
        .x_label_area_size(70)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..(results.len() as f64), 0.0..4.0)?;

    chart
        .configure_mesh()
        .x_desc("Subset Size (n)")
        .y_desc("d-Statistic Value")
        .axis_desc_style(("sans-serif", 26))
        .label_style(("sans-serif", 22))
        .x_labels(results.len())
        .x_label_formatter(&|x| {
            let idx = x.floor() as usize;
            if idx < results.len() && (x - (idx as f64) - 0.5).abs() < 0.1 {
                results[idx].label.clone()
            } else {
                String::new()
            }
        })
        .disable_x_mesh()
        .y_light_line_style(TRANSPARENT)
        .draw()?;

    let num_dashes = 60;
    let step = (results.len() as f64) / (num_dashes as f64);
    for d in (0..num_dashes).step_by(2) {
        let x0 = (d as f64) * step;
        let x1 = ((d + 1) as f64) * step;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, 2.0), (x1, 2.0)],
            RED.stroke_width(3),
        )))?;
    }

    let box_fill = RGBColor(166, 200, 224);
    let border_col = RGBColor(44, 62, 80);
    let mean_col = RGBColor(46, 204, 113);

    for (i, r) in results.iter().enumerate() {
        let center = (i as f64) + 0.5;
        let iqr = r.q3 - r.q1;
        let w_low = r.min_val.max(r.q1 - 1.5 * iqr);
        let w_high = r.max_val.min(r.q3 + 1.5 * iqr);

        let box_style = ShapeStyle {
            color: box_fill.to_rgba(),
            filled: true,
            stroke_width: 2,
        };
        chart.draw_series(std::iter::once(Rectangle::new(
            [(center - 0.22, r.q1), (center + 0.22, r.q3)],
            box_style,
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(center - 0.22, r.q1), (center + 0.22, r.q3)],
            border_col.stroke_width(2),
        )))?;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, r.q3), (center, w_high)],
            border_col.stroke_width(2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center - 0.11, w_high), (center + 0.11, w_high)],
            border_col.stroke_width(2),
        )))?;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, r.q1), (center, w_low)],
            border_col.stroke_width(2),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center - 0.11, w_low), (center + 0.11, w_low)],
            border_col.stroke_width(2),
        )))?;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center - 0.22, r.median), (center + 0.22, r.median)],
            BLACK.stroke_width(4),
        )))?;

        chart.draw_series(std::iter::once(TriangleMarker::new(
            (center, r.mean),
            10,
            mean_col.filled(),
        )))?;
    }

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, -1.0), (0.0, -1.0)],
            RED.stroke_width(3),
        )))?
        .label("No Autocorrelation (d = 2)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], RED.stroke_width(3)));

    chart
        .draw_series(std::iter::once(TriangleMarker::new(
            (0.0, -1.0),
            8,
            mean_col.filled(),
        )))?
        .label("Sample Mean")
        .legend(|(x, y)| TriangleMarker::new((x + 12, y), 9, mean_col.filled()));

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0.0, -1.0), (0.0, -1.0)],
            BLACK.stroke_width(4),
        )))?
        .label("Sample Median")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 25, y)], BLACK.stroke_width(4)));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 22))
        .draw()?;

    root.present()?;
    Ok(())
}