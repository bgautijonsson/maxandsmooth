#ifndef GEV_H
#define GEV_H

#include <RcppEigen.h>
#include <autodiff/forward/dual.hpp>
#include <Eigen/Dense>
#include <vector>

namespace gev {

    // Function to perform MLE for multiple locations
    Rcpp::List mle_multiple(Eigen::MatrixXd& data);

} // namespace gev

#endif // GEV_H
