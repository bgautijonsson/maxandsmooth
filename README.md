# maxandsmooth

This R package provides fast approximate Bayesian inference for latent Gaussian models using the Max-and-Smooth algorithm. The algorithm is implemented in C++ and uses Rcpp and RcppEigen to interface with R.

The code is split up into three main files:

* `src/max.cpp` implements the Max step of the algorithm, i.e. the maximum likelihood estimation. This file imports implemented distributions from other files.
* `src/smooth.cpp` implements the smoothing step of the algorithm, i.e. defining a latent gaussian field that takes in the maximum likelihood estimates and hessians from `src/max.cpp` and returns a smoothed estimate of the latent field, essentially treating the ML estimates as sufficient statistics for a latent gaussian field.
* `src/maxandsmooth.cpp` implements the Max-and-Smooth algorithm, tying together the maximum likelihood estimation and the smoothing step.

The main function for running the Max-and-Smooth algorithm is `maxandsmooth_cpp`. It takes a matrix of data, a family of distributions to fit, and uses a Gaussian Markov Random Field to smooth the maximum likelihood estimates using information from the hessians and the spatial structure of the data.

