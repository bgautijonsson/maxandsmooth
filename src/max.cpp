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
 * @return A list containing eta_hat, b_etay, and the 6 vectors of Hessian elements.
 */
Rcpp::List process_mle_results(Rcpp::List mle_results) {
    Eigen::MatrixXd mles = mle_results["mles"];
    Eigen::MatrixXd hessians = mle_results["hessians"];
    
    int n_loc = mles.rows();
    
    // 1. Reshape eta_hat into a vector of length 3n_loc
    Eigen::VectorXd eta_hat(3 * n_loc);
    eta_hat << mles.col(0), mles.col(1), mles.col(2);
    
    // 2. Create 6 vectors containing the elements of the hessians
    Eigen::VectorXd Q_psi_psi(n_loc), Q_tau_tau(n_loc), Q_phi_phi(n_loc);
    Eigen::VectorXd Q_psi_tau(n_loc), Q_psi_phi(n_loc), Q_tau_phi(n_loc);
    
    for (int i = 0; i < n_loc; ++i) {
        Q_psi_psi(i) = -hessians(i, 0);  // Negative because we want precision, not covariance
        Q_tau_tau(i) = -hessians(i, 4);
        Q_phi_phi(i) = -hessians(i, 8);
        Q_psi_tau(i) = -hessians(i, 1);  // or hessians(i, 3), they should be the same
        Q_psi_phi(i) = -hessians(i, 2);  // or hessians(i, 6)
        Q_tau_phi(i) = -hessians(i, 5);  // or hessians(i, 7)
    }
    
    // 3. Calculate b_etay as Q_etay times eta_hat
    Eigen::VectorXd b_etay(3 * n_loc);
    b_etay.segment(0, n_loc) = Q_psi_psi.cwiseProduct(eta_hat.segment(0, n_loc)) +
                                Q_psi_tau.cwiseProduct(eta_hat.segment(n_loc, n_loc)) +
                                Q_psi_phi.cwiseProduct(eta_hat.segment(2*n_loc, n_loc));
    b_etay.segment(n_loc, n_loc) = Q_psi_tau.cwiseProduct(eta_hat.segment(0, n_loc)) +
                                    Q_tau_tau.cwiseProduct(eta_hat.segment(n_loc, n_loc)) +
                                    Q_tau_phi.cwiseProduct(eta_hat.segment(2*n_loc, n_loc));
    b_etay.segment(2*n_loc, n_loc) = Q_psi_phi.cwiseProduct(eta_hat.segment(0, n_loc)) +
                                      Q_tau_phi.cwiseProduct(eta_hat.segment(n_loc, n_loc)) +
                                      Q_phi_phi.cwiseProduct(eta_hat.segment(2*n_loc, n_loc));
    
    // 4. Return the results
    return Rcpp::List::create(
        Rcpp::Named("eta_hat") = eta_hat,
        Rcpp::Named("b_etay") = b_etay,
        Rcpp::Named("Q_psi_psi") = Q_psi_psi,
        Rcpp::Named("Q_tau_tau") = Q_tau_tau,
        Rcpp::Named("Q_phi_phi") = Q_phi_phi,
        Rcpp::Named("Q_psi_tau") = Q_psi_tau,
        Rcpp::Named("Q_psi_phi") = Q_psi_phi,
        Rcpp::Named("Q_tau_phi") = Q_tau_phi
    );
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
    if (family == "gev") {
        Rcpp::List mle_results = gev::mle_multiple(data);
        return process_mle_results(mle_results);
    } else if (family == "gevt") {
        Rcpp::List mle_results = gevt::mle_multiple(data);
        return process_mle_results(mle_results);
    } else {
        stop("Invalid family");
    }
}
