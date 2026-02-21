# Deep Fréchet Regression
The code for the paper 'Deep Fréchet Regression'.

### Supporting software requirements

R version 4.4.0

Python version 3.11.4

### Libraries and dependencies used by the code

R packages to run Deep Frechet Regression methods:

* dplyr v 1.1.4
* tidyverse v 2.0.0
* frechet v 1.1.4
* fdadensity v 0.1.2
* vegan v 2.6-6.1
* Rtsne v 0.17
* uwot v 0.2.2
* diffusionMap v 1.2.0
* igraph v 2.0.3
* Matrix v 1.7-0
* geigen v 2.3
* osqp v 0.6.3.3
* MASS v 7.3-60.2
* matrixcalc v 1.0-6
* shapes 1.2.7
* doRNG v 1.8.6
* doParallel v 1.0.17
* ggplot2 v 3.5.1

R functions to run Global/Local Frechet Regression for distributional data

* https://github.com/yidongzhou/Wasserstein-regression-with-empirical-measures

R functions to run Global/Local Frechet Regression for network data

* https://github.com/yidongzhou/Network-Regression-with-Graph-Laplacians

R functions to run Sufficient Dimension Reduction for random objects

* https://github.com/bideliunian/DR4FrechetReg

Python modules to run Deep Frechet Regression methods:

* torch v 2.1.0
* numpy v 1.25.2
* pandas v 2.1.3
* scikit-learn v 1.3.2



### Folder Structure

* `./code`  code for all functions used in the paper.
* `./simulation/Distributional/`  code to reproduce simulations Section 5.2 and Section S.11, S.12 of the Supplementary Material for distributional data
* `./simulation/Network/`  code to reproduce simulations in Section S.10 of the Supplementary Material for network data.
* `./data/taxi/`  code to reproduce data analysis for New York Yellow Taxi network data in Section 6.
* `./data/mortality/`  code to reproduce data analysis for human mortality data in Section S.14 of the Supplementary Material.


### Reproducibility workflow

#### Simulations:

* Under `./simulation/distributional/`:
  
  + Run **`DenSimu.R`** to create Table 1 and Figure 2.
  
  + Run **`DenSimuManifold.R`** to create Table 4 and Table 5.

  + Run **`DenSimuRobust.R`** to create Table 6.

* Under `./simulation/network/`:
  
  + Run **`NetSimu.R`** to create Table 3 and Figure 5.


#### Data Applications:

* Under `./data/taxi/`, run **`taxi_app.R`** to do the analysis for New York Yellow Taxi network data and generate results in Section 6.

* Under `./data/mortality/`, run **`taxi_figure.R`** to do the analysis for New York Yellow Taxi network data and generate Figure 3 and Figure 4.

* Under `./data/mortality/`, run **`mortality_app.R`** to do the analysis for mortality data and generate Table 9.

* Under `./data/mortality/`, run **`mortality_figure.R`** to do the analysis for mortality data and generate Figure 6.
