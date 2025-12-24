## Official Implementations of `Adapting to Noise Tails in Private Linear Regression`

## Introduction

While the traditional goal of statistics is to learn about population parameters, modern practices have raised concerns about individuals and their privacy. One approach to address these concerns involves upgrading statistical methods into privacy preserving algorithms. In this paper, we develop a DP Huber estimator to enable privacy-preserving linear regression that is robust to heavy-tailed data. The trade-off among bias, privacy, and robustness is governed by a tunable robustification parameter. To privatize the DP Huber estimator, we implement noisy clipped gradient descent for low-dimensional settings and adopt noisy iterative hard thresholding for fitting high dimensional sparse models. Our method achieves near-optimal privacy-constrained convergence rates, up to logarithmic factors, under sub-Gaussian error conditions, relaxing several assumptions made in prior work. For heavy-tailed errors, we explicitly characterize the dependence of the non-asymptotic convergence rate on the moment index, privacy parameter, sample size, and intrinsic dimension. Our analysis reveals how the moment index influences robustification parameters, as well as the statistical error and privacy cost. By quantitatively exploring the interplay among bias, privacy, and robustness, we extend and complement classical perspectives on robustness and privacy. The proposed methods are further assessed on both simulated data and two real datasets.

## Data and Files

* **functions_main.py**: The "functions_main.py" file contains the Python implementation of Algorithms 1, 2, and 3 from the paper.

* **functions_for_confidenceinterval.py**: The "functions_for_confidenceinterval.py" file contains the Python implementation to obtain the covariance matrices, which are used to obtain the confidence intervals.

* **simulation_lowdim.ipynb**: The "simulation_lowdim.ipynb" file contains the notebook for Sections 5.2 (numerical studies in the low-dimensional setting).

* **simulation_highdim.ipynb**: The "simulation_highdim.ipynb" file contains the notebook for Sections 5.3 (numerical studies in the high-dimensional setting).

* **simulation_lowdim_GDP.ipynb**:  The "simulation_lowdim_GDP.ipynb" file contains the notebook for Section F.3 of the supplementary material (numerical studies under the GDP framework).

* **realdata.ipynb**: The "realdata.ipynb" file contains the notebook for Section A in the supplementary material (real data analysis) of the paper, including both low-dimensional and high-dimensional real data analysis.

* **realdata**: The "realdata" folder contains the real data used in "realdata.ipynb".

## Codes

The provided codes include the implementation of the estimation methods proposed in the paper (referred to as DP Huber and sparse DP Huber), along with their application to linear models.