#include <RcppEigen.h>
#include <string>
#include "max.h"
#include "smooth.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

Eigen::SparseMatrix<double> createQPrior(int x_dim, int y_dim) {
    int n = x_dim * y_dim;
    Eigen::SparseMatrix<double> Q_prior(n, n);
    Q_prior.reserve(Eigen::VectorXi::Constant(n, 5)); // Reserve space for 5 non-zeros per column

    // Create R matrix
    Eigen::SparseMatrix<double> R(x_dim, x_dim);
    R.reserve(Eigen::VectorXi::Constant(x_dim, 3)); // Reserve space for 3 non-zeros per column
    for (int i = 0; i < x_dim; ++i) {
        R.insert(i, i) = 2.0;
        if (i > 0) R.insert(i, i-1) = -1.0;
        if (i < x_dim - 1) R.insert(i, i+1) = -1.0;
    }
    R.makeCompressed();

    // Create I matrix
    Eigen::SparseMatrix<double> I(y_dim, y_dim);
    I.setIdentity();

    // Compute Q_prior = I ⊗ R + R ⊗ I
    Q_prior = Eigen::kroneckerProduct(I, R) + Eigen::kroneckerProduct(R, I);

    Q_prior.makeCompressed();
    return Q_prior;
}

Rcpp::List prepareLatentFieldParams(Rcpp::List max_results, int x_dim, int y_dim) {
    Eigen::VectorXd eta_hat = max_results["eta_hat"];
    Eigen::VectorXd b_etay = max_results["b_etay"];
    int n_locations = eta_hat.size() / 3;

    // Extract Q_etay components
    Eigen::VectorXd Q_psi_psi = max_results["Q_psi_psi"];
    Eigen::VectorXd Q_tau_tau = max_results["Q_tau_tau"];
    Eigen::VectorXd Q_phi_phi = max_results["Q_phi_phi"];
    Eigen::VectorXd Q_psi_tau = max_results["Q_psi_tau"];
    Eigen::VectorXd Q_psi_phi = max_results["Q_psi_phi"];
    Eigen::VectorXd Q_tau_phi = max_results["Q_tau_phi"];

    // Create Q_prior (this is a placeholder, you may need to adjust this)
    Eigen::SparseMatrix<double> Q_prior = createQPrior(x_dim, y_dim);

    // Initial values for tau (you may want to adjust these)
    Eigen::VectorXd tau_current = Eigen::VectorXd::Ones(3);

    return Rcpp::List::create(
        Rcpp::Named("n_blocks") = 3,
        Rcpp::Named("block_size") = n_locations,
        Rcpp::Named("eta_hat") = eta_hat,
        Rcpp::Named("Q_psi_psi") = Q_psi_psi,
        Rcpp::Named("Q_tau_tau") = Q_tau_tau,
        Rcpp::Named("Q_phi_phi") = Q_phi_phi,
        Rcpp::Named("Q_psi_tau") = Q_psi_tau,
        Rcpp::Named("Q_psi_phi") = Q_psi_phi,
        Rcpp::Named("Q_tau_phi") = Q_tau_phi,
        Rcpp::Named("b_etay") = b_etay,
        Rcpp::Named("Q_prior") = Q_prior,
        Rcpp::Named("tau_current") = tau_current
    );
}

// [[Rcpp::export]]
Rcpp::List maxandsmooth(
  Eigen::MatrixXd& data, 
  std::string& family,
  int x_dim,
  int y_dim,
  int n_iter, 
  int n_burnin, 
  int n_thin, 
  double proposal_sd
) {
    // Step 1: Perform Max step
    Rcpp::List max_results = ms_max(data, family);
    
    // Step 2: Prepare data for Smooth step
    Rcpp::List latent_field_params = prepareLatentFieldParams(max_results, x_dim, y_dim);
    
    // Step 3: Perform Smooth step
    Rcpp::List smooth_results = runSmooth(latent_field_params, n_iter, n_burnin, n_thin, proposal_sd);
    
    // Step 4: Combine results and return
    return Rcpp::List::create(
        Rcpp::Named("max_results") = max_results,
        Rcpp::Named("smooth_results") = smooth_results
    );
}


