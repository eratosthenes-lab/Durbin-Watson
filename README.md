# Durbin-Watson Statistic & Residual Ordering

This repository contains a Jupyter Notebook illustrating the sensitivity of the **Durbin-Watson ($d$) statistic** to the ordering of observations in regression analysis.

### Overview
The Durbin-Watson test is widely used to detect first-order autocorrelation in the residuals of a regression model:

$$d = \frac{\sum_{t=2}^{T} (e_t - e_{t-1})^2}{\sum_{t=1}^{T} e_t^2}$$

Because the numerator depends directly on the difference between consecutive residuals ($e_t - e_{t-1}$), sorting or rearranging the dataset changes the value of $d$, potentially leading to misleading conclusions if the underlying index (e.g., time or spatial sequence) is not properly preserved.

### Contents
* **Jupyter Notebook (`.ipynb`):** Step-by-step simulation, residual calculation, and comparison across different data orderings.

### Requirements
* Python 3.x
* `numpy`, `pandas`, `statsmodels`, `matplotlib`

### Author
**Matteo Putzulu** — [GitHub (@eratosthenes-lab)](https://github.com/eratosthenes-lab/Eratosthenes-Lab) | **LinkedIn:** [Matteo Putzulu](https://www.linkedin.com/in/matteoputzulu/) | **Medium:** [@mp.engineering](https://medium.com/@mp.engineering) | **ResearchGate:** [Matteo Putzulu](https://www.researchgate.net/profile/Matteo-Putzulu) | **Substack:** [The Quantitative Epoch](https://substack.com/@thequantitativeepoch) | **Academia.edu:** [Matteo Putzulu](https://independent.academia.edu/MatteoPutzulu) |


# Durbin-Watson - Monte Carlo Permutation Test (MCPT)

The purpose of this script is to highlight the importance of the “order” in which regression analysis residuals n are arranged, given that the value of d depends on the difference between consecutive residuals.
Simulations are performed using 10, 100, 1,000, ... up to 1 billion permutations of the n residuals. This approach applies the Monte Carlo method.
For each of the simulations, the d-value is calculated for every permutation, and the following statistical indices are reported: mean, median, standard deviation (SD), minimum, maximum, first quartile (Q1), and third quartile (Q3); the results are presented in a table and a boxplot to facilitate comparison.
In the script (.ipynb) "Durbin-Watson statistic and permutations", a fast single-step simulation will be performed in RAM, involving n elements (such as residuals), by following these straightforward steps implemented in Python; however, this limits the calculation to 10 million permutations (7 simulation from 10 up to 10 million permutations).
In the DWMCPT and DWPSD applications (.ipynb)—the latter of which features a GUI—the number of permutations reaches up to 1 billion.
To calculate d for each of the 9 simulations-which range from 10 up to 1 billion permutations-two strategies are used:
- for sample sizes up to 1 Million (1 M): fast single-step in-RAM;
- for large scales (10 M, 100 M, 1 B): leveraging high-capacity RAM.

For large scales the default batching size is 10 million for DWMCPT and 2 million for DWPSD; however, this value can be adjusted based on available RAM.
Thanks to an advanced graphical user interface (GUI), DWPSD allows for easy customization of input data:
- data source;
- centering relative to the mean;
- seed for random sampling;
- seed for simulation;
- batch size;

as well as the outputs:
- tables in CSV and Microsoft Excel XLSX formats;
- box plots (PNG, JPEG, PDF, SVG).

Python files in .ipynb, R file in .R, Julia file in .jl, C++ file in .cpp, Rust file in .rs, Go (Golang) file in .go, C# (.NET 8+) file in .cs, and CUDA / C++ (or Numba/CuPy) file for GPU in .cu.
#### See the PDF "DWPSD v.5.9" for details.
