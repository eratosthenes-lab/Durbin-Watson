// ==============================================================================
// DURBIN-WATSON MASSIVE SIMULATION - HIGH-PERFORMANCE GO (GOLANG) VERSION
// ==============================================================================

package main

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"math"
	mathrand "math/rand/v2"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
)

type SimStats struct {
	Label      string
	N          int
	RuntimeSec float64
	Mean       float64
	Median     float64
	MinVal     float64
	MaxVal     float64
	Q1         float64
	Q3         float64
	SD         float64
}

// 1. GLOBAL PROGRESS BAR FUNCTION
func printGlobalProgress(phaseName string, progressPercentage int, colorCode string) {
	const barLength = 30
	filledLength := int(math.Round(float64(barLength) * (float64(progressPercentage) / 100.0)))
	bar := strings.Repeat("█", filledLength) + strings.Repeat("-", barLength-filledLength)

	fmt.Printf("%s[GLOBAL PROGRESS] [%s] %3d%% | Phase: %s\033[0m\n",
		colorCode, bar, progressPercentage, phaseName)
}

// Inline Durbin-Watson statistic calculation on a permuted slice
func computeDWSingle(permVals []float64, invDenom float64) float64 {
	num := 0.0
	for j := 0; j < len(permVals)-1; j++ {
		d := permVals[j+1] - permVals[j]
		num += d * d
	}
	return num * invDenom
}

// Quickselect algorithm (O(N) average time) to find the k-th smallest element in-place
func quickSelect(data []float64, k int) float64 {
	left, right := 0, len(data)-1
	for left < right {
		pivotIdx := left + (right-left)/2
		pivotVal := data[pivotIdx]
		data[pivotIdx], data[right] = data[right], data[pivotIdx]

		storeIdx := left
		for i := left; i < right; i++ {
			if data[i] < pivotVal {
				data[storeIdx], data[i] = data[i], data[storeIdx]
				storeIdx++
			}
		}
		data[right], data[storeIdx] = data[storeIdx], data[right]

		if storeIdx == k {
			return data[k]
		} else if k < storeIdx {
			right = storeIdx - 1
		} else {
			left = storeIdx + 1
		}
	}
	return data[left]
}

// Fast linear-time quantile calculation
func computeQuantile(data []float64, q float64) float64 {
	n := len(data)
	pos := q * float64(n-1)
	idx := int(pos)
	frac := pos - float64(idx)

	val1 := quickSelect(data, idx)
	if frac == 0.0 || idx+1 >= n {
		return val1
	}

	// Minimum element in data[idx+1:]
	val2 := data[idx+1]
	for i := idx + 2; i < n; i++ {
		if data[i] < val2 {
			val2 = data[i]
		}
	}
	return val1 + frac*(val2-val1)
}

// Cryptographically random uint64 seed generator
func generateCryptoSeed() uint64 {
	var b [8]byte
	_, err := rand.Read(b[:])
	if err != nil {
		return uint64(time.Now().UnixNano())
	}
	return binary.LittleEndian.Uint64(b[:])
}

func main() {
	const (
		CYAN   = "\033[96m"
		YELLOW = "\033[93m"
		GREEN  = "\033[92m"
	)

	globalStart := time.Now()
	numWorkers := runtime.NumCPU()

	printGlobalProgress("Initialization", 0, CYAN)
	fmt.Printf("Global timer started successfully! (Active CPU cores: %d)\n", numWorkers)

	// 2. SEED CONFIGURATION (RANDOMLY GENERATED AND DISPLAYED)
	// Replace these two lines with fixed uint64 integers to reproduce a specific run
	sampleSeed := generateCryptoSeed()
	simulationSeed := generateCryptoSeed()

	fmt.Printf("\n[SEED INFO] Assigned SAMPLE_SEED:     %d\n", sampleSeed)
	fmt.Printf("[SEED INFO] Assigned SIMULATION_SEED: %d\n\n", simulationSeed)

	// 3. SIMULATION PARAMETERS
	const sampleSize = 50
	rngSample := mathrand.New(mathrand.NewPCG(sampleSeed, sampleSeed^0x9e3779b97f4a7c15))

	values := make([]float64, sampleSize)
	denominator := 0.0
	for i := 0; i < sampleSize; i++ {
		values[i] = rngSample.NormFloat64()
		denominator += values[i] * values[i]
	}
	invDenominator := 1.0 / denominator

	targetSimulations := []int{
		10, 100, 1_000, 10_000, 100_000,
		1_000_000, 10_000_000, 100_000_000, 1_000_000_000,
	}
	labels := []string{"10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B"}
	colHeaders := []string{"10", "100", "1,000", "10,000", "100,000", "1 M", "10 M", "100 M", "1 B"}

	results := make([]SimStats, 0, len(targetSimulations))

	// 4. RUN SIMULATIONS (PHASE 1/2)
	printGlobalProgress("Simulation Run (Phase 1/2)", 10, YELLOW)
	fmt.Println("Starting massive simulation up to 1 Billion...\n")

	for idx, n := range targetSimulations {
		cellStart := time.Now()

		globalPct := 10 + int((float64(idx)/float64(len(targetSimulations)))*60.0)
		printGlobalProgress(fmt.Sprintf("Subset n = %s", colHeaders[idx]), globalPct, YELLOW)

		dwValues := make([]float64, n)

		if n <= 10_000_000 {
			// Deterministic single-goroutine execution for small/medium n
			simRNG := mathrand.New(mathrand.NewPCG(simulationSeed, simulationSeed^0xbf58476d1ce4e5b9))
			buf := slices.Clone(values)
			for i := 0; i < n; i++ {
				simRNG.Shuffle(len(buf), func(a, b int) { buf[a], buf[b] = buf[b], buf[a] })
				dwValues[i] = computeDWSingle(buf, invDenominator)
			}
		} else {
			// Batched & Multi-goroutine execution for 100M and 1B
			const batchSize = 10_000_000
			numBatches := n / batchSize
			stepPrint := max(1, numBatches/20)

			fmt.Printf("Processing n = %s in %d batches across %d goroutines:\n",
				colHeaders[idx], numBatches, numWorkers)

			// Derive deterministic sub-seeds from simulationSeed
			masterRNG := mathrand.New(mathrand.NewPCG(simulationSeed, simulationSeed^0x94d049bb133111eb))
			batchSeeds := make([]uint64, numBatches)
			for b := 0; b < numBatches; b++ {
				batchSeeds[b] = masterRNG.Uint64()
			}

			for b := 0; b < numBatches; b++ {
				startIdx := b * batchSize
				endIdx := startIdx + batchSize

				var wg sync.WaitGroup
				chunkLen := (batchSize + numWorkers - 1) / numWorkers

				for w := 0; w < numWorkers; w++ {
					tStart := startIdx + w*chunkLen
					tEnd := min(endIdx, tStart+chunkLen)
					if tStart >= tEnd {
						continue
					}

					wg.Add(1)
					go func(workerID, sIdx, eIdx int, bSeed uint64) {
						defer wg.Done()
						seed1 := bSeed + uint64(workerID)*0x9e3779b97f4a7c15
						seed2 := bSeed ^ (uint64(workerID) + 1)
						localRNG := mathrand.New(mathrand.NewPCG(seed1, seed2))
						localBuf := slices.Clone(values)

						for i := sIdx; i < eIdx; i++ {
							localRNG.Shuffle(len(localBuf), func(a, b int) {
								localBuf[a], localBuf[b] = localBuf[b], localBuf[a]
							})
							dwValues[i] = computeDWSingle(localBuf, invDenominator)
						}
					}(w, tStart, tEnd, batchSeeds[b])
				}
				wg.Wait()

				if (b+1)%stepPrint == 0 || (b+1) == numBatches {
					progress := float64(b+1) / float64(numBatches)
					elapsed := time.Since(cellStart).Seconds()
					eta := 0.0
					if progress < 1.0 {
						eta = (elapsed / progress) - elapsed
					}

					filled := int(math.Round(20.0 * progress))
					pbar := strings.Repeat("=", filled) + strings.Repeat("-", 20-filled)

					fmt.Printf("\r[%s] %3d%% | Batch %d/%d | ETA: %dm %ds   ",
						pbar, int(progress*100), b+1, numBatches, int(eta)/60, int(eta)%60)
				}
			}
			fmt.Println()
		}

		// Parallel Mean, SD, Min, Max computation across workers
		type partialStats struct {
			sum    float64
			minVal float64
			maxVal float64
		}
		partials := make([]partialStats, numWorkers)
		var wgStats sync.WaitGroup
		chunk := (n + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			s := w * chunk
			e := min(n, s+chunk)
			if s >= e {
				partials[w] = partialStats{0, math.Inf(1), math.Inf(-1)}
				continue
			}
			wgStats.Add(1)
			go func(wID, start, end int) {
				defer wgStats.Done()
				localSum := 0.0
				localMin := dwValues[start]
				localMax := dwValues[start]
				for i := start; i < end; i++ {
					v := dwValues[i]
					localSum += v
					if v < localMin {
						localMin = v
					}
					if v > localMax {
						localMax = v
					}
				}
				partials[wID] = partialStats{localSum, localMin, localMax}
			}(w, s, e)
		}
		wgStats.Wait()

		totalSum := 0.0
		minVal := math.Inf(1)
		maxVal := math.Inf(-1)
		for _, p := range partials {
			totalSum += p.sum
			if p.minVal < minVal {
				minVal = p.minVal
			}
			if p.maxVal > maxVal {
				maxVal = p.maxVal
			}
		}
		meanVal := totalSum / float64(n)

		// Parallel Standard Deviation
		sqDiffPartials := make([]float64, numWorkers)
		for w := 0; w < numWorkers; w++ {
			s := w * chunk
			e := min(n, s+chunk)
			if s >= e {
				continue
			}
			wgStats.Add(1)
			go func(wID, start, end int) {
				defer wgStats.Done()
				localSq := 0.0
				for i := start; i < end; i++ {
					d := dwValues[i] - meanVal
					localSq += d * d
				}
				sqDiffPartials[wID] = localSq
			}(w, s, e)
		}
		wgStats.Wait()

		totalSqDiff := 0.0
		for _, sq := range sqDiffPartials {
			totalSqDiff += sq
		}
		sdVal := math.Sqrt(totalSqDiff / float64(n-1))

		// Linear-time quantiles (Q1, Median, Q3)
		q1Val := computeQuantile(dwValues, 0.25)
		medVal := computeQuantile(dwValues, 0.50)
		q3Val := computeQuantile(dwValues, 0.75)

		runtimeSec := time.Since(cellStart).Seconds()

		results = append(results, SimStats{
			Label:      labels[idx],
			N:          n,
			RuntimeSec: runtimeSec,
			Mean:       meanVal,
			Median:     medVal,
			MinVal:     minVal,
			MaxVal:     maxVal,
			Q1:         q1Val,
			Q3:         q3Val,
			SD:         sdVal,
		})

		fmt.Printf("✓ Completed n = %s in %.2f s\n\n", colHeaders[idx], runtimeSec)
	}

	// 5. PRINT SUMMARY TABLE & EXPORT CSV
	fmt.Printf("\n%-14s", "Index")
	for _, h := range colHeaders {
		fmt.Printf("%12s", h)
	}
	fmt.Println("\n" + strings.Repeat("-", 14+12*len(colHeaders)))

	printRow := func(name string, getter func(SimStats) float64, format string) {
		fmt.Printf("%-14s", name)
		for _, r := range results {
			fmt.Printf("%12s", fmt.Sprintf(format, getter(r)))
		}
		fmt.Println()
	}

	printRow("Runtime (s)", func(s SimStats) float64 { return s.RuntimeSec }, "%.2f")
	printRow("Mean", func(s SimStats) float64 { return s.Mean }, "%.5f")
	printRow("Median", func(s SimStats) float64 { return s.Median }, "%.5f")
	printRow("Min", func(s SimStats) float64 { return s.MinVal }, "%.5f")
	printRow("Max", func(s SimStats) float64 { return s.MaxVal }, "%.5f")
	printRow("Q1", func(s SimStats) float64 { return s.Q1 }, "%.5f")
	printRow("Q3", func(s SimStats) float64 { return s.Q3 }, "%.5f")
	printRow("SD", func(s SimStats) float64 { return s.SD }, "%.5f")

	csvFilename := "durbin_watson_simulation_results_go.csv"
	csvFile, err := os.Create(csvFilename)
	if err != nil {
		fmt.Printf("[ERROR] Failed to create CSV: %v\n", err)
		return
	}
	writer := csv.NewWriter(csvFile)

	headerRow := append([]string{"Index"}, colHeaders...)
	_ = writer.Write(headerRow)

	writeCSVRow := func(name string, getter func(SimStats) float64, format string) {
		row := make([]string, 0, len(results)+1)
		row = append(row, name)
		for _, r := range results {
			row = append(row, fmt.Sprintf(format, getter(r)))
		}
		_ = writer.Write(row)
	}

	writeCSVRow("Runtime (s)", func(s SimStats) float64 { return s.RuntimeSec }, "%.2f")
	writeCSVRow("Mean", func(s SimStats) float64 { return s.Mean }, "%.5f")
	writeCSVRow("Median", func(s SimStats) float64 { return s.Median }, "%.5f")
	writeCSVRow("Min", func(s SimStats) float64 { return s.MinVal }, "%.5f")
	writeCSVRow("Max", func(s SimStats) float64 { return s.MaxVal }, "%.5f")
	writeCSVRow("Q1", func(s SimStats) float64 { return s.Q1 }, "%.5f")
	writeCSVRow("Q3", func(s SimStats) float64 { return s.Q3 }, "%.5f")
	writeCSVRow("SD", func(s SimStats) float64 { return s.SD }, "%.5f")

	writer.Flush()
	_ = csvFile.Close()

	fmt.Printf("\n[SUCCESS] Matrix data exported to '%s'\n", csvFilename)
	printGlobalProgress("Simulation Completed", 75, YELLOW)

	// 6. BOXPLOT DATA & GNUPLOT SCRIPT GENERATION (PHASE 2/2)
	printGlobalProgress("Generating Boxplots (Phase 2/2)", 80, GREEN)

	datFilename := "dw_boxplot_stats_go.dat"
	datFile, err := os.Create(datFilename)
	if err == nil {
		_, _ = fmt.Fprintln(datFile, "# idx label whisker_low q1 median q3 whisker_high mean")
		for i, r := range results {
			iqr := r.Q3 - r.Q1
			wLow := max(r.MinVal, r.Q1-1.5*iqr)
			wHigh := min(r.MaxVal, r.Q3+1.5*iqr)
			_, _ = fmt.Fprintf(datFile, "%d \"%s\" %.6f %.6f %.6f %.6f %.6f %.6f\n",
				i+1, r.Label, wLow, r.Q1, r.Median, r.Q3, wHigh, r.Mean)
		}
		_ = datFile.Close()
	}

	gpFilename := "plot_dw_go.gp"
	gpFile, err := os.Create(gpFilename)
	if err == nil {
		_, _ = fmt.Fprintf(gpFile, `set terminal pngcairo size 2000,1000 enhanced font 'Arial,14'
set output 'dw_simulation_boxplots_go.png'
set title 'Durbin-Watson Statistic Distribution with Mean and Median (Up to 1 Billion)' font 'Arial,18'
set xlabel 'Subset Size (n)' font 'Arial,15'
set ylabel 'd-Statistic Value' font 'Arial,15'
set yrange [0:4]
set xrange [0.3:%.1f]
set grid ytics lt 0 lw 1 lc rgb '#888888'
set key top right box opaque
set boxwidth 0.45
set style fill solid 0.85 border lc rgb '#2c3e50'
set arrow from 0.3,2.0 to %.1f,2.0 nohead dt 2 lw 2 lc rgb 'red'
plot '%s' using 1:4:3:7:6:xtic(2) with candlesticks lc rgb '#2c3e50' lw 1.5 fc rgb '#a6c8e0' title 'IQR & 1.5x Whiskers' whiskerbars 0.5, \
     '' using 1:5:5:5:5 with candlesticks lc rgb '#000000' lw 2.5 title 'Sample Median', \
     '' using 1:8 with points pt 9 ps 1.8 lc rgb '#2ecc71' title 'Sample Mean', \
     NaN with lines dt 2 lw 2 lc rgb 'red' title 'No Autocorrelation (d = 2)'
`, float64(len(results))+0.7, float64(len(results))+0.7, datFilename)
		_ = gpFile.Close()
	}

	cmd := exec.Command("gnuplot", gpFilename)
	if err := cmd.Run(); err == nil {
		fmt.Println("[SUCCESS] Boxplot saved to 'dw_simulation_boxplots_go.png'")
	} else {
		fmt.Printf("[INFO] Gnuplot script saved to '%s'. Install gnuplot and run 'gnuplot %s' to render the PNG.\n", gpFilename, gpFilename)
	}

	printGlobalProgress("Execution Finished", 100, GREEN)

	// 7. TOTAL RUNTIME
	totalElapsed := time.Since(globalStart).Seconds()
	fmt.Printf("\n[TIMER] Total execution time: %.2f seconds (%dm %ds)\n",
		totalElapsed, int(totalElapsed)/60, int(totalElapsed)%60)
}