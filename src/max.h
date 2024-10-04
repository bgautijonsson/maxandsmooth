#ifndef MAX_H
#define MAX_H

#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// Function declaration for max_cpp
Rcpp::List max_cpp(const Eigen::MatrixXd& data, const std::string& family);

#endif // MAX_H
