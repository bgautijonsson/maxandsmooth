#include "LatentField.h"
#include <RcppEigen.h>


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(Rcpp)]]

LatentField::LatentField(
    int n_blocks, 
    int block_size,
    const Eigen::VectorXd& eta_hat, 
    const Eigen::VectorXd& Q_psi_psi,
    const Eigen::VectorXd& Q_tau_tau,
    const Eigen::VectorXd& Q_phi_phi,
    const Eigen::VectorXd& Q_psi_tau,
    const Eigen::VectorXd& Q_psi_phi,
    const Eigen::VectorXd& Q_tau_phi,
    const Eigen::VectorXd& b_etay,
    const Eigen::SparseMatrix<double>& Q_prior,
    const Eigen::VectorXd& tau_current
) : eta_hat_(eta_hat), 
    Q_psi_psi_(Q_psi_psi.asDiagonal().toDenseMatrix().sparseView()),
    Q_tau_tau_(Q_tau_tau.asDiagonal().toDenseMatrix().sparseView()),
    Q_phi_phi_(Q_phi_phi.asDiagonal().toDenseMatrix().sparseView()),
    Q_psi_tau_(Q_psi_tau.asDiagonal().toDenseMatrix().sparseView()),
    Q_psi_phi_(Q_psi_phi.asDiagonal().toDenseMatrix().sparseView()),
    Q_tau_phi_(Q_tau_phi.asDiagonal().toDenseMatrix().sparseView()),
    b_etay_(b_etay),
    Q_prior_(Q_prior),
    n_blocks_(n_blocks),
    block_size_(block_size),
    L_post_current_(n_blocks, block_size),
    L_post_proposed_(n_blocks, block_size),
    tau_current_(tau_current)
{
    L_post_current_.setPrecisionBlocks(
        Q_psi_psi_, 
        Q_tau_tau_, 
        Q_phi_phi_, 
        Q_psi_tau_, 
        Q_psi_phi_, 
        Q_tau_phi_, 
        Q_prior_
    );
    L_post_current_.setTau(tau_current_);
    L_post_current_.computeDecomposition();
    y_current_ = L_post_current_.forwardSolve(b_etay_);
    mu_post_current_ = L_post_current_.backwardSolve(y_current_);
}

Eigen::VectorXd LatentField::getCurrentTau() const {
    return tau_current_;
}

int LatentField::getTauSize() const {
    return n_blocks_;
}

int LatentField::getEtaSize() const {
    return block_size_ * n_blocks_;
}

void LatentField::UpdateProposals(const Eigen::VectorXd& tau_prop) {
    tau_proposed_ = tau_prop;
    L_post_proposed_.setPrecisionBlocks(
        Q_psi_psi_, 
        Q_tau_tau_, 
        Q_phi_phi_, 
        Q_psi_tau_, 
        Q_psi_phi_, 
        Q_tau_phi_, 
        Q_prior_
    );
    L_post_proposed_.setTau(tau_proposed_);
    L_post_proposed_.computeDecomposition();
    y_proposed_ = L_post_proposed_.forwardSolve(b_etay_);
}

bool LatentField::AcceptReject(double log_prior_ratio) {
    double log_latent_cond_prior = 0;
    double log_latent_cond_data = 0;
    for (int i = 0; i < n_blocks_; i++) {
        log_latent_cond_prior += std::log(tau_proposed_[i]) - std::log(tau_current_[i]);
    }

    log_latent_cond_prior *= block_size_ / 2;
    log_latent_cond_data += L_post_proposed_.logDeterminant() - L_post_current_.logDeterminant();
    log_latent_cond_data -= (y_proposed_.squaredNorm() - y_current_.squaredNorm()) / 2;

    double log_accept_prob = log_prior_ratio + log_latent_cond_prior + log_latent_cond_data;
    double u = R::runif(0, 1);
    if (log(u) < log_accept_prob) {
        return true;
    } else {
        return false;
    }
}

void LatentField::UpdateCurrent() {
    tau_current_ = tau_proposed_;
    L_post_current_ = L_post_proposed_;
    y_current_ = y_proposed_;
    mu_post_current_ = L_post_current_.backwardSolve(y_current_);
}

Eigen::VectorXd LatentField::sampleLatentField() {
    Eigen::VectorXd eta_sample;
    Rcpp::NumericVector z = Rcpp::rnorm(block_size_ * n_blocks_, 0, 1);
    Eigen::Map<Eigen::VectorXd> z_eigen(z.begin(), z.size());
    eta_sample = mu_post_current_ + L_post_current_.forwardSolve(z_eigen);
    return eta_sample;
}
