#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <RcppEigen.h>


// [[Rcpp::depends(RcppEigen)]]
using namespace autodiff;
using namespace Eigen;


/**
 * @brief Computes the simultaneous log-likelihood of the GEV distribution for all locations.
 */
var loglik(const ArrayXvar& x, const Eigen::MatrixXd& data) {

  int N = data.rows();
  int P = data.cols();
  var loglik = 0;

  for (int p = 0; p < P; ++p) {
    var mu = exp(x[p]);
    var sigma = exp(x[p] + x[P + p]);
    var xi = pow(1 + exp(-x[2 * P + p]), -1);

    for(int i = 0; i < N; ++i) {
      var z = (data(i) - mu) / sigma;

      if (xi < 1e-6) {
          loglik -= log(sigma) + z + exp(-z);
      } else {
          var t = 1 + xi * z;
          loglik -= log(sigma) + (1.0 + 1.0 / xi) * log(t) + exp(-1.0 / xi * log(t));
      }
    }
  }


  return loglik;
}

// [[Rcpp::export]]
Eigen::VectorXd gradient(Eigen::MatrixXd& data) {

  int P = data.cols();
  VectorXvar x(3 * P);
  for (int p = 0; p < P; ++p) {
    x[p] = 0;
    x[P + p] = 0;
    x[2 * P + p] = 0;
  }

  var y = loglik(x, data);
  Eigen::VectorXd dydx = gradient(y, x);
  
  return dydx;
}


