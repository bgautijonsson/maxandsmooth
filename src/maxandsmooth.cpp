#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::MatrixXd exampleFunction(const Eigen::MatrixXd& X) {
    // Your code here
    return X;
}
