#include <RcppEigen.h>
#include "gev.h"
#include <string>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

/**
 * @brief Performs the Max step of Max & Smooth: computes Maximum Likelihood Estimates for multiple locations in parallel.
 *
 * @param data A matrix where each column represents data for a location.
 * @param family The distribution family: "gev" for the GEV distribution.
 * @return A matrix of MLEs with each row corresponding to a location's parameters.
 */
// [[Rcpp::export]]
Rcpp::List max(Eigen::MatrixXd& data, std::string family) {
  int n_locations = data.cols();
  if (family == "gev") {
    return gev::mle_multiple(data);
  } else {
    stop("Invalid family");
  }
}