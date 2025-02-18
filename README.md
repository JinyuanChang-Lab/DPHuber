## Official Implementations of `Adapting to Noise Tails in Private Linear Regression`

## Introduction

While the traditional goal of statistics is to learn about population parameters, modern practices have raised concerns about individuals and their privacy. One approach to address these concerns involves upgrading statistical methods into privacy preserving algorithms. In this paper, we develop a DP Huber estimator to enable privacy-preserving linear regression that is robust to heavy-tailed data. The trade-off among bias, privacy, and robustness is governed by a tunable robustification parameter. To privatize the DP Huber estimator, we implement noisy clipped gradient descent for low-dimensional settings and adopt noisy iterative hard thresholding for fitting high dimensional sparse models. Our method achieves near-optimal privacy-constrained convergence rates, up to logarithmic factors, under sub-Gaussian error conditions, relaxing several assumptions made in prior work. For heavy-tailed errors, we explicitly
characterize the dependence of the non-asymptotic convergence rate on the moment index, privacy parameter, sample size, and intrinsic dimension. Our analysis reveals how the moment index influences robustification parameters, as well as the statistical error and privacy cost. By quantitatively exploring the interplay among bias, privacy, and robustness, we extend and complement classical perspectives on robustness and privacy. The proposed methods are further assessed on both simulated data and two real datasets.

## Data and Files

* **final_util.py**: The "final_util.py" file contains the Python implementation of Algorithms 1, 2, and 3 from the paper.

* **final_lowdim.ipynb**: The notebook file "final_lowdim.ipynb" contains the codes for Section 5.1 (numerical studies in the low-dimensional setting) of the paper, including data generation and multiple repeated experiments.

* **final_highdim.ipynb**: The notebook file "final_highdim.ipynb" contains the codes for Section 5.2 (numerical studies in the high-dimensional setting) of the paper, including data generation and multiple repeated experiments.

* **final_realdata.ipynb**: The notebook file "final_realdata.ipynb" contains the codes for Section 6 (real data analysis) of the paper, including both low-dimensional and high-dimensional real data analysis.

* **realdata**: The "realdata" folder contains the real data used in "final_realdata.ipynb".

## Codes

The provided codes include the implementation of the estimation methods proposed in the paper (referred to as DP Huber and sparse DP Huber), along with their application to linear models.
