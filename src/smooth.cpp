// Smooth.cpp
#include <RcppEigen.h>
#include "LatentField.h"
#include <random>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(Rcpp)]]

class Smooth {
private:
    LatentField latent_field_;
    int n_iter_;
    int n_burnin_;
    int n_thin_;
    
    // Hyperparameters for proposal distribution
    double proposal_sd_;
    
    // Store MCMC samples
    Eigen::MatrixXd tau_samples_;
    Eigen::MatrixXd eta_samples_;

    // Helper methods
    Eigen::VectorXd proposeTau(const Eigen::VectorXd& current_tau);
    double logPriorTau(const Eigen::VectorXd& tau);

public:
    Smooth(
      LatentField&& latent_field, 
      int n_iter, 
      int n_burnin, 
      int n_thin, 
      double proposal_sd
    ) : latent_field_(std::move(latent_field)), 
        n_iter_(n_iter), 
        n_burnin_(n_burnin), 
        n_thin_(n_thin), 
        proposal_sd_(proposal_sd) {
          tau_samples_ = Eigen::MatrixXd::Zero(n_iter_ / n_thin_, latent_field_.getTauSize());
          eta_samples_ = Eigen::MatrixXd::Zero(n_iter_ / n_thin_, latent_field_.getEtaSize());
        }

    void run();
    Rcpp::List getSamples();
};

void Smooth::run() {
    Eigen::VectorXd current_tau = latent_field_.getCurrentTau();
    int current_sample = 0;
    Eigen::VectorXd eta_sample;
    for (int i = 0; i < n_iter_ + n_burnin_; ++i) {
        // Propose new tau
        Eigen::VectorXd proposed_tau = proposeTau(current_tau);
        
        // Update latent field with proposed tau
        latent_field_.UpdateProposals(proposed_tau);
        
        // Compute log prior ratio
        double log_prior_ratio = logPriorTau(proposed_tau) - logPriorTau(current_tau);
        
        // Accept/Reject step
        if (latent_field_.AcceptReject(log_prior_ratio)) {
            current_tau = proposed_tau;
            latent_field_.UpdateCurrent();
        }
        
        // Sample new eta
        eta_sample = latent_field_.sampleLatentField();
        
        // Store samples (accounting for burn-in and thinning)
        if (i >= n_burnin_ && (i - n_burnin_) % n_thin_ == 0) {
            tau_samples_.row(current_sample) = current_tau;
            eta_samples_.row(current_sample) = eta_sample;
            current_sample++;
        }

      if (i % 100 == 0) {
        Rcpp::Rcout << "Iteration: " << i << std::endl;
      }
    }
}

Eigen::VectorXd Smooth::proposeTau(const Eigen::VectorXd& current_tau) {
    Eigen::VectorXd log_proposed_tau = current_tau.array().log();
    double proposal_step;
    for (int i = 0; i < current_tau.size(); ++i) {
        proposal_step = R::rnorm(0, proposal_sd_);
        log_proposed_tau[i] += proposal_step;
    }
    return log_proposed_tau.array().exp();
}

double Smooth::logPriorTau(const Eigen::VectorXd& tau) {
    // Implement log prior for tau
    // For example, assuming independent gamma priors:
    double log_prior = 0.0;
    double alpha = 1.0; // shape parameter
    double beta = 1.0;  // rate parameter
    for (int i = 0; i < tau.size(); ++i) {
        log_prior += (alpha - 1) * std::log(tau[i]) - beta * tau[i];
    }
    return log_prior;
}

Rcpp::List Smooth::getSamples() {
    return Rcpp::List::create(
        Rcpp::Named("tau") = tau_samples_,
        Rcpp::Named("eta") = eta_samples_
    );
}

Rcpp::List runSmooth(Rcpp::List latent_field_params, int n_iter, int n_burnin, int n_thin, double proposal_sd) {
    int n_blocks = Rcpp::as<int>(latent_field_params["n_blocks"]);
    int block_size = Rcpp::as<int>(latent_field_params["block_size"]);
    Eigen::VectorXd eta_hat = Rcpp::as<Eigen::VectorXd>(latent_field_params["eta_hat"]);
    Eigen::VectorXd Q_psi_psi = Rcpp::as<Eigen::VectorXd>(latent_field_params["Q_psi_psi"]);
    Eigen::VectorXd Q_tau_tau = Rcpp::as<Eigen::VectorXd>(latent_field_params["Q_tau_tau"]);
    Eigen::VectorXd Q_phi_phi = Rcpp::as<Eigen::VectorXd>(latent_field_params["Q_phi_phi"]);
    Eigen::VectorXd Q_psi_tau = Rcpp::as<Eigen::VectorXd>(latent_field_params["Q_psi_tau"]);
    Eigen::VectorXd Q_psi_phi = Rcpp::as<Eigen::VectorXd>(latent_field_params["Q_psi_phi"]);
    Eigen::VectorXd Q_tau_phi = Rcpp::as<Eigen::VectorXd>(latent_field_params["Q_tau_phi"]);
    Eigen::VectorXd b_etay = Rcpp::as<Eigen::VectorXd>(latent_field_params["b_etay"]);
    Eigen::SparseMatrix<double> Q_prior = Rcpp::as<Eigen::SparseMatrix<double>>(latent_field_params["Q_prior"]);
    Eigen::VectorXd tau_current = Rcpp::as<Eigen::VectorXd>(latent_field_params["tau_current"]);

    LatentField latent_field(
      n_blocks, 
      block_size,
      eta_hat, 
      Q_psi_psi,
      Q_tau_tau,
      Q_phi_phi,
      Q_psi_tau,
      Q_psi_phi,
      Q_tau_phi,
      b_etay,
      Q_prior,
      tau_current
    );
    Smooth smoother(std::move(latent_field), n_iter, n_burnin, n_thin, proposal_sd);
    smoother.run();
    return smoother.getSamples();
}