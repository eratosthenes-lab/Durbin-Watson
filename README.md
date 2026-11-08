# Durbin-Watson
The purpose of this script is to highlight the importance of the “order” in which regression analysis residuals n are arranged, given that the value of d depends on the difference between consecutive residuals.
Simulations are performed using 10, 100, 1,000, ... up to 1 billion permutations of the n residuals. This approach applies the Monte Carlo method.
For each of the simulations, the d-value is calculated for every permutation, and the following statistical indices are reported: mean, median, standard deviation (SD), minimum, maximum, first quartile (Q1), and third quartile (Q3); the results are presented in a table and a boxplot to facilitate comparison.
In the script (.ipynb) "Durbin-Watson statistic and permutations", a fast single-step simulation will be performed in RAM, involving n elements (such as residuals), by following these straightforward steps implemented in Python; however, this limits the calculation to 10 million permutations (7 simulation from 10 up to 10 million permutations).
In the DWMCPT and DWPSD applications (.ipynb)—the latter of which features a GUI—the number of permutations reaches up to 1 billion.
To calculate d for each of the 9 simulations-which range from 10 up to 1 billion permutations-two strategies are used:
- for sample sizes up to 1 Million (1 M): fast single-step in-RAM;
- for large scales (10 M, 100 M, 1 B): leveraging high-capacity RAM.

for large scales the default batching size is 10 million for DWMCPT and 2 million for DWPSD; however, this value can, however, be adjusted based on available RAM.
Thanks to an advanced graphical user interface (GUI), DWPSD allows for easy customization of input data:
- data source
- centering relative to the mean
- seed for random sampling
- seed for simulation
- batch size

as well as the outputs:
- tables in CSV and Excel formats
- box plots (PNG, JPEG, PDF, SVG)
