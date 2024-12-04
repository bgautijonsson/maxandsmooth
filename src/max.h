#ifndef MAX_H
#define MAX_H

#include <RcppEigen.h>
#include <string>

// Function declarations
Rcpp::List process_mle_results(Rcpp::List mle_results);
Rcpp::List ms_max(Eigen::MatrixXd& data, std::string& family);

#endif // MAX_H

