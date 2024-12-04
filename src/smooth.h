#ifndef SMOOTH_H
#define SMOOTH_H

#include <RcppEigen.h>
#include "LatentField.h"

// [[Rcpp::depends(RcppEigen)]]

Rcpp::List runSmooth(Rcpp::List latent_field_params, int n_iter, int n_burnin, int n_thin, double proposal_sd);

#endif // SMOOTH_H

