#include <RcppEigen.h>
#include <omp.h>
#include "gev.h"
#include "max.h"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

// Parallelized function to apply MLE to each column of a matrix
// [[Rcpp::export]]
Rcpp::List max_cpp(const Eigen::MatrixXd& data, const std::string& family) {
  int n_stations = data.cols();
  Eigen::MatrixXd estimates(n_stations, 3);
  Rcpp::List hessians;

  if (family == "gev") {
    hessians = Rcpp::List(n_stations);

    #pragma omp parallel for
    for (int i = 0; i < n_stations; ++i) {
      Eigen::VectorXd station_data = data.col(i);
      GEV::MLEResult result = GEV::mle(station_data);
      
      #pragma omp critical
      {
        estimates.row(i) = result.estimates;
        
        hessians[i] = result.hessian;
        
      }
    }
  } else {
    throw std::runtime_error("Unsupported family");
  }

  Rcpp::List output;
  output["estimates"] = estimates;
  output["hessians"] = hessians;

  return output;
}
