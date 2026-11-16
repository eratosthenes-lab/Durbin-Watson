// ==============================================================================
// DURBIN-WATSON MASSIVE SIMULATION - HIGH-PERFORMANCE C# (.NET 8+) VERSION
// ==============================================================================

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace DurbinWatsonSim
{
    public sealed class SimStats
    {
        public string Label { get; init; } = "";
        public int N { get; init; }
        public double RuntimeSec { get; init; }
        public double Mean { get; init; }
        public double Median { get; init; }
        public double MinVal { get; init; }
        public double MaxVal { get; init; }
        public double Q1 { get; init; }
        public double Q3 { get; init; }
        public double SD { get; init; }
    }

    // Fast deterministic Xoshiro256** PRNG with Box-Muller normal generation
    public struct Xoshiro256StarStar
    {
        private ulong _s0, _s1, _s2, _s3;

        public Xoshiro256StarStar(ulong seed)
        {
            _s0 = SplitMix64(ref seed);
            _s1 = SplitMix64(ref seed);
            _s2 = SplitMix64(ref seed);
            _s3 = SplitMix64(ref seed);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong SplitMix64(ref ulong x)
        {
            ulong z = (x += 0x9E3779B97F4A7C15UL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong NextUInt64()
        {
            ulong result = BitOperations.RotateLeft(_s1 * 5UL, 7) * 9UL;
            ulong t = _s1 << 17;

            _s2 ^= _s0;
            _s3 ^= _s1;
            _s1 ^= _s2;
            _s0 ^= _s3;

            _s2 ^= t;
            _s3 = BitOperations.RotateLeft(_s3, 45);

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double NextDouble()
        {
            return (NextUInt64() >> 11) * (1.0 / (1UL << 53));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int NextInt(int maxExclusive)
        {
            return (int)(((ulong)(uint)maxExclusive * (NextUInt64() >> 32)) >> 32);
        }

        public double NextGaussian()
        {
            double u1 = Math.Max(NextDouble(), double.Epsilon);
            double u2 = NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Shuffle(Span<double> span)
        {
            for (int i = span.Length - 1; i > 0; i--)
            {
                int j = NextInt(i + 1);
                (span[i], span[j]) = (span[j], span[i]);
            }
        }
    }

    internal static class Program
    {
        // 1. GLOBAL PROGRESS BAR FUNCTION
        private static void PrintGlobalProgress(string phaseName, int progressPercentage, string colorCode)
        {
            const int barLength = 30;
            int filledLength = (int)Math.Round(barLength * (progressPercentage / 100.0));
            string bar = new string('█', filledLength) + new string('-', barLength - filledLength);

            Console.WriteLine($"{colorCode}[GLOBAL PROGRESS] [{bar}] {progressPercentage,3}% | Phase: {phaseName}\u001b[0m");
            Console.Out.Flush();
        }

        // Hardware-accelerated SIMD (AVX2 / Vector256) Durbin-Watson calculation
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ComputeDWSingleSIMD(ReadOnlySpan<double> permVals, double invDenom)
        {
            double num = 0.0;
            int len = permVals.Length - 1;
            int j = 0;

            if (Vector256.IsHardwareAccelerated && len >= Vector256<double>.Count)
            {
                Vector256<double> vSum = Vector256<double>.Zero;
                int limit = len - Vector256<double>.Count;

                for (; j <= limit; j += Vector256<double>.Count)
                {
                    Vector256<double> vCurr = Vector256.Create(permVals.Slice(j, Vector256<double>.Count));
                    Vector256<double> vNext = Vector256.Create(permVals.Slice(j + 1, Vector256<double>.Count));
                    Vector256<double> vDiff = vNext - vCurr;
                    vSum += vDiff * vDiff;
                }
                num = Vector256.Sum(vSum);
            }

            for (; j < len; j++)
            {
                double d = permVals[j + 1] - permVals[j];
                num += d * d;
            }

            return num * invDenom;
        }

        // Quickselect algorithm (O(N) average time) to find the k-th smallest element in-place
        private static double QuickSelect(Span<double> data, int k)
        {
            int left = 0;
            int right = data.Length - 1;

            while (left < right)
            {
                int pivotIdx = left + ((right - left) >> 1);
                double pivotVal = data[pivotIdx];
                (data[pivotIdx], data[right]) = (data[right], data[pivotIdx]);

                int storeIdx = left;
                for (int i = left; i < right; i++)
                {
                    if (data[i] < pivotVal)
                    {
                        (data[storeIdx], data[i]) = (data[i], data[storeIdx]);
                        storeIdx++;
                    }
                }
                (data[right], data[storeIdx]) = (data[storeIdx], data[right]);

                if (storeIdx == k)
                    return data[k];
                if (k < storeIdx)
                    right = storeIdx - 1;
                else
                    left = storeIdx + 1;
            }

            return data[left];
        }

        // Fast linear-time quantile calculation
        private static double ComputeQuantile(Span<double> data, double q)
        {
            int n = data.Length;
            double pos = q * (n - 1);
            int idx = (int)pos;
            double frac = pos - idx;

            double val1 = QuickSelect(data, idx);
            if (frac == 0.0 || idx + 1 >= n)
                return val1;

            double val2 = data[idx + 1];
            for (int i = idx + 2; i < n; i++)
            {
                if (data[i] < val2)
                    val2 = data[i];
            }
            return val1 + frac * (val2 - val1);
        }

        private static ulong GenerateCryptoSeed()
        {
            Span<byte> bytes = stackalloc byte[8];
            RandomNumberGenerator.Fill(bytes);
            return BitConverter.ToUInt64(bytes);
        }

        private static void Main()
        {
            const string CYAN = "\u001b[96m";
            const string YELLOW = "\u001b[93m";
            const string GREEN = "\u001b[92m";

            var globalStopwatch = Stopwatch.StartNew();
            int numThreads = Environment.ProcessorCount;

            PrintGlobalProgress("Initialization", 0, CYAN);
            Console.WriteLine($"Global timer started successfully! (Active CPU cores: {numThreads} | SIMD Vector256: {Vector256.IsHardwareAccelerated})");

            // 2. SEED CONFIGURATION (RANDOMLY GENERATED AND DISPLAYED)
            // Replace these two lines with fixed ulong integers to reproduce a specific run
            ulong sampleSeed = GenerateCryptoSeed();
            ulong simulationSeed = GenerateCryptoSeed();

            Console.WriteLine($"\n[SEED INFO] Assigned SAMPLE_SEED:     {sampleSeed}");
            Console.WriteLine($"[SEED INFO] Assigned SIMULATION_SEED: {simulationSeed}\n");

            // 3. SIMULATION PARAMETERS
            const int sampleSize = 50;
            var rngSample = new Xoshiro256StarStar(sampleSeed);

            double[] values = new double[sampleSize];
            double denominator = 0.0;
            for (int i = 0; i < sampleSize; i++)
            {
                values[i] = rngSample.NextGaussian();
                denominator += values[i] * values[i];
            }
            double invDenominator = 1.0 / denominator;

            int[] targetSimulations =
            {
                10, 100, 1_000, 10_000, 100_000,
                1_000_000, 10_000_000, 100_000_000, 1_000_000_000
            };
            string[] labels = { "10", "100", "1K", "10K", "100K", "1M", "10M", "100M", "1B" };
            string[] colHeaders = { "10", "100", "1,000", "10,000", "100,000", "1 M", "10 M", "100 M", "1 B" };

            var results = new List<SimStats>(targetSimulations.Length);

            // 4. RUN SIMULATIONS (PHASE 1/2)
            PrintGlobalProgress("Simulation Run (Phase 1/2)", 10, YELLOW);
            Console.WriteLine("Starting massive simulation up to 1 Billion...\n");

            for (int idx = 0; idx < targetSimulations.Length; idx++)
            {
                int n = targetSimulations[idx];
                var cellStopwatch = Stopwatch.StartNew();

                int globalPct = 10 + (int)((idx / (double)targetSimulations.Length) * 60.0);
                PrintGlobalProgress($"Subset n = {colHeaders[idx]}", globalPct, YELLOW);

                double[] dwValues = new double[n];

                if (n <= 10_000_000)
                {
                    // Deterministic single-thread execution for small/medium n
                    var simRng = new Xoshiro256StarStar(simulationSeed);
                    Span<double> buf = stackalloc double[sampleSize];
                    values.CopyTo(buf);

                    for (int i = 0; i < n; i++)
                    {
                        simRng.Shuffle(buf);
                        dwValues[i] = ComputeDWSingleSIMD(buf, invDenominator);
                    }
                }
                else
                {
                    // Batched & Multi-threaded Parallel.For execution for 100M and 1B
                    const int batchSize = 10_000_000;
                    int numBatches = n / batchSize;
                    int stepPrint = Math.Max(1, numBatches / 20);

                    Console.WriteLine($"Processing n = {colHeaders[idx]} in {numBatches} batches across {numThreads} threads:");

                    // Derive deterministic sub-seeds from simulationSeed
                    var masterRng = new Xoshiro256StarStar(simulationSeed);
                    ulong[] batchSeeds = new ulong[numBatches];
                    for (int b = 0; b < numBatches; b++)
                    {
                        batchSeeds[b] = masterRng.NextUInt64();
                    }

                    for (int b = 0; b < numBatches; b++)
                    {
                        int startIdx = b * batchSize;
                        int endIdx = startIdx + batchSize;
                        ulong currentBatchSeed = batchSeeds[b];

                        // Partition batch across available CPU cores with thread-local stack buffers
                        Parallel.For(0, numThreads, tid =>
                        {
                            int chunkLen = (batchSize + numThreads - 1) / numThreads;
                            int tStart = startIdx + tid * chunkLen;
                            int tEnd = Math.Min(endIdx, tStart + chunkLen);

                            if (tStart < tEnd)
                            {
                                ulong seed = currentBatchSeed + unchecked((ulong)tid * 0x9E3779B97F4A7C15UL);
                                var localRng = new Xoshiro256StarStar(seed);
                                Span<double> localBuf = stackalloc double[sampleSize];
                                values.CopyTo(localBuf);

                                for (int i = tStart; i < tEnd; i++)
                                {
                                    localRng.Shuffle(localBuf);
                                    dwValues[i] = ComputeDWSingleSIMD(localBuf, invDenominator);
                                }
                            }
                        });

                        if ((b + 1) % stepPrint == 0 || (b + 1) == numBatches)
                        {
                            double progress = (b + 1) / (double)numBatches;
                            double elapsed = cellStopwatch.Elapsed.TotalSeconds;
                            double eta = progress < 1.0 ? (elapsed / progress) - elapsed : 0.0;

                            int filled = (int)Math.Round(20.0 * progress);
                            string pbar = new string('=', filled) + new string('-', 20 - filled);

                            Console.Write($"\r[{pbar}] {(int)(progress * 100),3}% | Batch {b + 1}/{numBatches} | ETA: {(int)eta / 60}m {(int)eta % 60}s   ");
                        }
                    }
                    Console.WriteLine();
                }

                // Parallel Mean, Min, Max computation across CPU cores
                object lockObj = new();
                double totalSum = 0.0;
                double minVal = double.PositiveInfinity;
                double maxVal = double.NegativeInfinity;

                int statChunk = (n + numThreads - 1) / numThreads;
                Parallel.For(0, numThreads, tid =>
                {
                    int s = tid * statChunk;
                    int e = Math.Min(n, s + statChunk);
                    if (s >= e) return;

                    double localSum = 0.0;
                    double localMin = dwValues[s];
                    double localMax = dwValues[s];

                    for (int i = s; i < e; i++)
                    {
                        double v = dwValues[i];
                        localSum += v;
                        if (v < localMin) localMin = v;
                        if (v > localMax) localMax = v;
                    }

                    lock (lockObj)
                    {
                        totalSum += localSum;
                        if (localMin < minVal) minVal = localMin;
                        if (localMax > maxVal) maxVal = localMax;
                    }
                });

                double meanVal = totalSum / n;

                // Parallel Standard Deviation
                double totalSqDiff = 0.0;
                Parallel.For(0, numThreads, tid =>
                {
                    int s = tid * statChunk;
                    int e = Math.Min(n, s + statChunk);
                    if (s >= e) return;

                    double localSq = 0.0;
                    for (int i = s; i < e; i++)
                    {
                        double d = dwValues[i] - meanVal;
                        localSq += d * d;
                    }

                    lock (lockObj)
                    {
                        totalSqDiff += localSq;
                    }
                });

                double sdVal = Math.Sqrt(totalSqDiff / (n - 1));

                // Linear-time in-place quantiles (Q1, Median, Q3)
                double q1Val = ComputeQuantile(dwValues, 0.25);
                double medVal = ComputeQuantile(dwValues, 0.50);
                double q3Val = ComputeQuantile(dwValues, 0.75);

                double runtimeSec = cellStopwatch.Elapsed.TotalSeconds;

                results.Add(new SimStats
                {
                    Label = labels[idx],
                    N = n,
                    RuntimeSec = runtimeSec,
                    Mean = meanVal,
                    Median = medVal,
                    MinVal = minVal,
                    MaxVal = maxVal,
                    Q1 = q1Val,
                    Q3 = q3Val,
                    SD = sdVal
                });

                Console.WriteLine($"✓ Completed n = {colHeaders[idx]} in {runtimeSec:F2} s\n");
            }

            // 5. PRINT SUMMARY TABLE & EXPORT CSV
            Console.Write($"\n{"Index",-14}");
            foreach (string h in colHeaders)
                Console.Write($"{h,12}");
            Console.WriteLine("\n" + new string('-', 14 + 12 * colHeaders.Length));

            void PrintRow(string name, Func<SimStats, double> getter, string format)
            {
                Console.Write($"{name,-14}");
                foreach (var r in results)
                    Console.Write($"{getter(r).ToString(format, CultureInfo.InvariantCulture),12}");
                Console.WriteLine();
            }

            PrintRow("Runtime (s)", s => s.RuntimeSec, "F2");
            PrintRow("Mean", s => s.Mean, "F5");
            PrintRow("Median", s => s.Median, "F5");
            PrintRow("Min", s => s.MinVal, "F5");
            PrintRow("Max", s => s.MaxVal, "F5");
            PrintRow("Q1", s => s.Q1, "F5");
            PrintRow("Q3", s => s.Q3, "F5");
            PrintRow("SD", s => s.SD, "F5");

            const string csvFilename = "durbin_watson_simulation_results_csharp.csv";
            using (var csvWriter = new StreamWriter(csvFilename))
            {
                csvWriter.Write("Index");
                foreach (string h in colHeaders)
                    csvWriter.Write($",\"{h}\"");
                csvWriter.WriteLine();

                void WriteCsvRow(string name, Func<SimStats, double> getter, string format)
                {
                    csvWriter.Write(name);
                    foreach (var r in results)
                        csvWriter.Write($",{getter(r).ToString(format, CultureInfo.InvariantCulture)}");
                    csvWriter.WriteLine();
                }

                WriteCsvRow("Runtime (s)", s => s.RuntimeSec, "F2");
                WriteCsvRow("Mean", s => s.Mean, "F5");
                WriteCsvRow("Median", s => s.Median, "F5");
                WriteCsvRow("Min", s => s.MinVal, "F5");
                WriteCsvRow("Max", s => s.MaxVal, "F5");
                WriteCsvRow("Q1", s => s.Q1, "F5");
                WriteCsvRow("Q3", s => s.Q3, "F5");
                WriteCsvRow("SD", s => s.SD, "F5");
            }

            Console.WriteLine($"\n[SUCCESS] Matrix data exported to '{csvFilename}'");
            PrintGlobalProgress("Simulation Completed", 75, YELLOW);

            // 6. BOXPLOT DATA & GNUPLOT SCRIPT GENERATION (PHASE 2/2)
            PrintGlobalProgress("Generating Boxplots (Phase 2/2)", 80, GREEN);

            const string datFilename = "dw_boxplot_stats_csharp.dat";
            using (var datWriter = new StreamWriter(datFilename))
            {
                datWriter.WriteLine("# idx label whisker_low q1 median q3 whisker_high mean");
                for (int i = 0; i < results.Count; i++)
                {
                    var r = results[i];
                    double iqr = r.Q3 - r.Q1;
                    double wLow = Math.Max(r.MinVal, r.Q1 - 1.5 * iqr);
                    double wHigh = Math.Min(r.MaxVal, r.Q3 + 1.5 * iqr);

                    datWriter.WriteLine(string.Format(
                        CultureInfo.InvariantCulture,
                        "{0} \"{1}\" {2:F6} {3:F6} {4:F6} {5:F6} {6:F6} {7:F6}",
                        i + 1, r.Label, wLow, r.Q1, r.Median, r.Q3, wHigh, r.Mean));
                }
            }

            const string gpFilename = "plot_dw_csharp.gp";
            using (var gpWriter = new StreamWriter(gpFilename))
            {
                double xMax = results.Count + 0.7;
                gpWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
@"set terminal pngcairo size 2000,1000 enhanced font 'Arial,14'
set output 'dw_simulation_boxplots_csharp.png'
set title 'Durbin-Watson Statistic Distribution with Mean and Median (Up to 1 Billion)' font 'Arial,18'
set xlabel 'Subset Size (n)' font 'Arial,15'
set ylabel 'd-Statistic Value' font 'Arial,15'
set yrange [0:4]
set xrange [0.3:{0:F1}]
set grid ytics lt 0 lw 1 lc rgb '#888888'
set key top right box opaque
set boxwidth 0.45
set style fill solid 0.85 border lc rgb '#2c3e50'
set arrow from 0.3,2.0 to {0:F1},2.0 nohead dt 2 lw 2 lc rgb 'red'
plot '{1}' using 1:4:3:7:6:xtic(2) with candlesticks lc rgb '#2c3e50' lw 1.5 fc rgb '#a6c8e0' title 'IQR & 1.5x Whiskers' whiskerbars 0.5, \
     '' using 1:5:5:5:5 with candlesticks lc rgb '#000000' lw 2.5 title 'Sample Median', \
     '' using 1:8 with points pt 9 ps 1.8 lc rgb '#2ecc71' title 'Sample Mean', \
     NaN with lines dt 2 lw 2 lc rgb 'red' title 'No Autocorrelation (d = 2)'",
                    xMax, datFilename));
            }

            try
            {
                var psi = new ProcessStartInfo("gnuplot", gpFilename)
                {
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using var proc = Process.Start(psi);
                proc?.WaitForExit();
                if (proc?.ExitCode == 0)
                    Console.WriteLine("[SUCCESS] Boxplot saved to 'dw_simulation_boxplots_csharp.png'");
                else
                    Console.WriteLine($"[INFO] Gnuplot script saved to '{gpFilename}'. Run 'gnuplot {gpFilename}' to render the PNG.");
            }
            catch
            {
                Console.WriteLine($"[INFO] Gnuplot script saved to '{gpFilename}'. Install gnuplot and run 'gnuplot {gpFilename}' to render the PNG.");
            }

            PrintGlobalProgress("Execution Finished", 100, GREEN);

            // 7. TOTAL RUNTIME
            double totalElapsed = globalStopwatch.Elapsed.TotalSeconds;
            Console.WriteLine($"\n[TIMER] Total execution time: {totalElapsed:F2} seconds ({(int)totalElapsed / 60}m {(int)totalElapsed % 60}s)");
        }
    }
}