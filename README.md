# E2M: End-to-End Deep Learning for Predicting Metric Space-Valued Outputs

The code for the paper 'E2M: End-to-End Deep Learning for Predicting Metric Space-Valued Outputs'.

### Supporting software requirements

R version 4.4.0

### Libraries and dependencies used by the code

R packages to run E2M and the comparative methods include:

* reticulate v 1.38.0
* dplyr v 1.1.4
* tidyverse v 2.0.0
* frechet v 1.1.4
* fdadensity v 0.1.2
* KScorrect v 1.4.0
* EnvStats v 3.0.0
* Matrix v 1.7-0
* geigen v 2.3
* osqp v 0.6.3.3
* MASS v 7.3-60.2
* matrixcalc v 1.0-6
* shapes 1.2.7
* trust v 0.1-8
* pracma v 2.4.4
* purrr v 1.0.2
* manifold v 0.1.1
* foreach v 1.5.2
* doRNG v 1.8.6
* doParallel v 1.0.17
* stats v 4.4.0
* vegan v 2.6-6.1
* ggplot2 v 3.5.1

### Folder Structure

* `./code/E2M` code for all functions used in the paper.
* `./code/DFR` R functions to run Deep Fréchet Regression.
* `./code/DR4FrechetReg` R functions to run Sufficient Dimension Reduction for non-Euclidean responses.
* `./code/Network-Regression-with-Graph-Laplacians` R functions to run Global Fréchet Regression for network data.
* `./code/Single-Index-Frechet` R functions to run Single Index Fréchet Regression.
* `./code/Wasserstein-regression-with-empirical-measures` R functions to run Global Fréchet Regression for distributional data.
* `./code/Code_RFWLFR` R functions to run Random Forest Weighted Local Fréchet Regression.
* `./Application/Mortality/` code to reproduce data analysis for human mortality data in Section 6.1.
* `./Application/Taxi/` code to reproduce data analysis for New York yellow taxi network data in Section 6.2.
* `./Simulation/DenSimu.R` code to reproduce simulations in Section 6.2 for distributional data
* `./Simulation/DenSimu_scale.R` code to reproduce simulations in Section 6.4 for distributional data
* `./Simulation/DenSimu_linear.R` code to reproduce simulations in Appendix G for distributional data
* `./Simulation/NetSimu.R` code to reproduce simulations in Section 6.2 for network data.
* `./Simulation/NetSimu_scale.R` code to reproduce simulations in Section 6.4 for network data.
* `./Simulation/SPDSimu.R` code to reproduce simulations in Section 6.2 for SPD matrix data with power metric.
* `./Simulation/SPDSimu_BW.R` code to reproduce simulations in Section 6.2 for SPD matrix data with Bures-Wasserstein metric.

