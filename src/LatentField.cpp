#include <Eigen/Dense>
#include "block_cholesky.h"

class LatentField {
public:
    LatentField(const Eigen::VectorXd& eta_hat, 
                const Eigen::MatrixXd& Q_eta_y, 
                const Eigen::MatrixXd& Q_prior,
                int n_blocks, 
                int block_size)
        : eta_hat_(eta_hat), 
          Q_eta_y_(Q_eta_y),
          Q_prior_(Q_prior),
          cholesky_current_(n_blocks, block_size),
          cholesky_proposed_(n_blocks, block_size),
          block_size_(block_size)
    {
        // Precompute b_etay
        b_etay_ = Q_eta_y_ * eta_hat_;
    }

    void updateCurrentPrecision(const Eigen::VectorXd& tau_eta) {
        Eigen::MatrixXd Q_post = Q_eta_y_;
        for (int i = 0; i < tau_eta.size(); ++i) {
            Q_post.block(i*block_size_, i*block_size_, block_size_, block_size_) += tau_eta[i] * Q_prior_;
        }
        cholesky_current_.setPrecisionMatrix(Q_post);
        cholesky_current_.computeDecomposition();
        
        y_current_ = cholesky_current_.forwardSolve(b_etay_);
        mu_post_current_ = cholesky_current_.backwardSolve(y_current_);
        log_det_current_ = cholesky_current_.logDeterminant();
    }

    void updateProposedPrecision(const Eigen::VectorXd& tau_eta) {
        Eigen::MatrixXd Q_post = Q_eta_y_;
        for (int i = 0; i < tau_eta.size(); ++i) {
            Q_post.block(i*block_size_, i*block_size_, block_size_, block_size_) += tau_eta[i] * Q_prior_;
        }
        cholesky_proposed_.setPrecisionMatrix(Q_post);
        cholesky_proposed_.computeDecomposition();
        
        y_proposed_ = cholesky_proposed_.forwardSolve(b_etay_);
        log_det_proposed_ = cholesky_proposed_.logDeterminant();
    }

    void acceptProposal() {
        std::swap(cholesky_current_, cholesky_proposed_);
        y_current_ = y_proposed_;
        mu_post_current_ = cholesky_current_.backwardSolve(y_current_);
        log_det_current_ = log_det_proposed_;
    }

    Eigen::VectorXd sampleEta() {
        Eigen::VectorXd z = Eigen::VectorXd::Random(eta_hat_.size());
        return mu_post_current_ + cholesky_current_.solve(z);
    }

    double getLogLikelihood(const Eigen::VectorXd& eta) const {
        return -0.5 * log_det_current_ 
               - 0.5 * (eta - eta_hat_).transpose() * Q_eta_y_ * (eta - eta_hat_);
    }

    double getProposedLogLikelihood(const Eigen::VectorXd& eta) const {
        return -0.5 * log_det_proposed_ 
               - 0.5 * (eta - eta_hat_).transpose() * Q_eta_y_ * (eta - eta_hat_);
    }

    double getCurrentLogDeterminant() const { return log_det_current_; }
    double getProposedLogDeterminant() const { return log_det_proposed_; }

private:
    Eigen::VectorXd eta_hat_;
    Eigen::MatrixXd Q_eta_y_;
    Eigen::MatrixXd Q_prior_;
    Eigen::VectorXd b_etay_;  // Precomputed Q_eta_y * eta_hat
    
    block_cholesky::BlockCholesky cholesky_current_;
    block_cholesky::BlockCholesky cholesky_proposed_;
    
    Eigen::VectorXd y_current_;
    Eigen::VectorXd mu_post_current_;
    double log_det_current_;
    
    Eigen::VectorXd y_proposed_;
    double log_det_proposed_;
    
    int block_size_;
};