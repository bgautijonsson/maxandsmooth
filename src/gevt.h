#ifndef GEVT_H
#define GEVT_H

#include <RcppEigen.h>
#include <autodiff/forward/dual.hpp>
#include <Eigen/Dense>
#include <vector>

namespace gevt {

    // Function to perform MLE for multiple locations
    Rcpp::List mle_multiple(Eigen::MatrixXd& data);

} // namespace gevt

#endif // GEVT_H
