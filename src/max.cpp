#include <RcppEigen.h>
#include "gev.h"
#include "gevt.h"
#include <string>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

/**
 * @brief Processes the results from mle_multiple to prepare for the Smooth step.
 *
 * @param mle_results A list containing the MLEs and Hessians from mle_multiple.
 * @param num_params The number of parameters for the distribution family.
 * @return A list containing eta_hat and the vectors of Hessian elements.
 */
Rcpp::List process_mle_results(Rcpp::List mle_results, int num_params) {
    Eigen::MatrixXd mles = mle_results["mles"];
    Eigen::MatrixXd hessians = mle_results["hessians"];
    
    int n_loc = mles.rows();
    
    // 1. Reshape eta_hat into a vector of length num_params * n_loc
    Eigen::VectorXd eta_hat(num_params * n_loc);
    for (int i = 0; i < num_params; ++i) {
        eta_hat.segment(i * n_loc, n_loc) = mles.col(i);
    }
    
    // 2. Create vectors containing the elements of the hessians
    std::vector<Eigen::VectorXd> Q_elements(num_params * (num_params + 1) / 2, Eigen::VectorXd(n_loc));
    
    int index = 0;
    for (int i = 0; i < num_params; ++i) {
        for (int j = i; j < num_params; ++j) {
            for (int k = 0; k < n_loc; ++k) {
                Q_elements[index](k) = -hessians(k, i * num_params + j);
            }
            index++;
        }
    }
    
    // 4. Return the results
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("eta_hat") = eta_hat);
    for (int i = 0; i < Q_elements.size(); ++i) {
        result.push_back(Q_elements[i], "Q_" + std::to_string(i));
    }
    return result;
}

/**
 * @brief Performs the Max step of Max & Smooth: computes Maximum Likelihood Estimates for multiple locations in parallel.
 *
 * @param data A matrix where each column represents data for a location.
 * @param family The distribution family: "gev" for the GEV distribution.
 * @return A list containing the processed MLE results.
 */
// [[Rcpp::export]]
Rcpp::List ms_max(Eigen::MatrixXd& data, std::string& family) {
    int n_locations = data.cols();
    int num_params;
    Rcpp::List mle_results;
    
    if (family == "gev") {
        num_params = 3;
        mle_results = gev::mle_multiple(data);
    } else if (family == "gevt") {
        num_params = 4;
        mle_results = gevt::mle_multiple(data);
    } else {
        stop("Invalid family");
    }
    
    return process_mle_results(mle_results, num_params);
}
