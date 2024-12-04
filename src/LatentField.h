#ifndef LATENT_FIELD_H
#define LATENT_FIELD_H

#include <RcppEigen.h>
#include "block_cholesky.h"

class LatentField {
public:
    LatentField(
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
    );

    void UpdateProposals(const Eigen::VectorXd& tau_prop);
    bool AcceptReject(double log_prior_ratio);
    void UpdateCurrent();
    Eigen::VectorXd sampleLatentField();
    Eigen::VectorXd getCurrentTau() const;
    int getTauSize() const;
    int getEtaSize() const;

private:
    int n_blocks_;
    int block_size_;

    Eigen::VectorXd eta_hat_;

    Eigen::SparseMatrix<double> Q_psi_psi_;
    Eigen::SparseMatrix<double> Q_tau_tau_;
    Eigen::SparseMatrix<double> Q_phi_phi_;
    Eigen::SparseMatrix<double> Q_psi_tau_;
    Eigen::SparseMatrix<double> Q_psi_phi_;
    Eigen::SparseMatrix<double> Q_tau_phi_;

    Eigen::VectorXd b_etay_;

    Eigen::SparseMatrix<double> Q_prior_;

    block_cholesky::BlockCholesky L_post_current_;
    Eigen::VectorXd tau_current_;
    Eigen::VectorXd y_current_;
    Eigen::VectorXd mu_post_current_;

    block_cholesky::BlockCholesky L_post_proposed_;
    Eigen::VectorXd tau_proposed_;
    Eigen::VectorXd y_proposed_;
};

#endif // LATENT_FIELD_H
