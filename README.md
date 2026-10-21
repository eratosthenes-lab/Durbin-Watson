# Durbin-Watson
The purpose of this script is to highlight the importance of the “order” in which regression analysis residuals are arranged, given that the value of d depends on the difference between consecutive residuals. To this end, a simple simulation involving n residuals will be performed, following these simple steps, implemented in Python:
1 - random sample of n residuals and calculation of its permutations (n!);
2- calculation of the Durbin-Watson d statistic for this sample;
3- random sampling of 7subsets (simulations) of 10, 100, 1,000, 10,000, 100,000, 1,000,000, and 10,000,000 permutations from the set of permutations (n!);
4- for each of the 7 samples, calculate the d for each permutation and report the following statistical indices: mean, median, standard deviation (SD), minimum, maximum, first quartile (Q1) and third quartile (Q3);
5- organize the outputs in a table to compare them.
See: https://medium.com/@mp.engineering/durbin-watson-statistic-and-permutations-of-residuals-93db39a54d66
