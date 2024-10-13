#include "block_cholesky.h"
#include <Eigen/Cholesky>
#include <stdexcept>
#include <cmath>

namespace block_cholesky {

BlockCholesky::BlockCholesky(int n_blocks, int block_size)
    : n_blocks_(n_blocks), block_size_(block_size), log_det_(0.0) {
    // Initialize precision matrices as zero matrices
    Q_pp_ = Eigen::MatrixXd::Zero(block_size_, block_size_);
    Q_tt_ = Eigen::MatrixXd::Zero(block_size_, block_size_);
    Q_ff_ = Eigen::MatrixXd::Zero(block_size_, block_size_);
    Q_pt_ = Eigen::MatrixXd::Zero(block_size_, block_size_);
    Q_pf_ = Eigen::MatrixXd::Zero(block_size_, block_size_);
    Q_tf_ = Eigen::MatrixXd::Zero(block_size_, block_size_);
}

void BlockCholesky::setPrecisionBlocks(const Eigen::MatrixXd& Q_pp, 
                                      const Eigen::MatrixXd& Q_tt, 
                                      const Eigen::MatrixXd& Q_ff,
                                      const Eigen::MatrixXd& Q_pt, 
                                      const Eigen::MatrixXd& Q_pf, 
                                      const Eigen::MatrixXd& Q_tf) {
    if (Q_pp.rows() != block_size_ || Q_pp.cols() != block_size_ ||
        Q_tt.rows() != block_size_ || Q_tt.cols() != block_size_ ||
        Q_ff.rows() != block_size_ || Q_ff.cols() != block_size_ ||
        Q_pt.rows() != block_size_ || Q_pt.cols() != block_size_ ||
        Q_pf.rows() != block_size_ || Q_pf.cols() != block_size_ ||
        Q_tf.rows() != block_size_ || Q_tf.cols() != block_size_) {
        throw std::invalid_argument("Block size mismatch in setPrecisionBlocks.");
    }
    Q_pp_ = Q_pp;
    Q_tt_ = Q_tt;
    Q_ff_ = Q_ff;
    Q_pt_ = Q_pt;
    Q_pf_ = Q_pf;
    Q_tf_ = Q_tf;
}

Eigen::MatrixXd BlockCholesky::choleskyDecompose(const Eigen::MatrixXd& block) const {
    Eigen::LLT<Eigen::MatrixXd> llt(block);
    if (llt.info() == Eigen::NumericalIssue) {
        // Matrix is not positive definite
        return Eigen::MatrixXd();
    }
    return llt.matrixL();
}

bool BlockCholesky::computeDecomposition() {
    // Step 1: Compute L11 = Cholesky(Q_pp)
    L11_ = choleskyDecompose(Q_pp_);
    if (L11_.size() == 0) {
        return false; // Decomposition failed
    }
    // Accumulate log determinant: 2 * sum(log(diagonal elements of L11))
    log_det_ += 2.0 * L11_.diagonal().array().log().sum();

    // Step 2: Compute L21 = Q_pt * L11^{-T}
    Eigen::MatrixXd L11_invT = L11_.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(block_size_, block_size_));
    L21_ = Q_pt_ * L11_invT;

    // Step 3: Compute L31 = Q_pf * L11^{-T}
    L31_ = Q_pf_ * L11_invT;

    // Step 4: Compute S22 = Q_tt + tau_tau Q_prior - L21 * L21^T
    Eigen::MatrixXd S22 = Q_tt_ - L21_ * L21_.transpose();
    L22_ = choleskyDecompose(S22);
    if (L22_.size() == 0) {
        return false; // Decomposition failed
    }
    // Accumulate log determinant
    log_det_ += 2.0 * L22_.diagonal().array().log().sum();

    // Step 5: Compute L32 = (Q_tf - L31 * L21^T) * L22^{-T}
    Eigen::MatrixXd L22_invT = L22_.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(block_size_, block_size_));
    L32_ = (Q_tf_ - L31_ * L21_.transpose()) * L22_invT;

    // Step 6: Compute S33 = Q_ff + tau_phi Q_prior - L31 * L31^T - L32 * L32^T
    Eigen::MatrixXd S33 = Q_ff_ - L31_ * L31_.transpose() - L32_ * L32_.transpose();
    L33_ = choleskyDecompose(S33);
    if (L33_.size() == 0) {
        return false; // Decomposition failed
    }
    // Accumulate log determinant
    log_det_ += 2.0 * L33_.diagonal().array().log().sum();

    return true;
}

Eigen::VectorXd BlockCholesky::forwardSolve(const Eigen::VectorXd& b) const {
    if (L11_.size() == 0 || L22_.size() == 0 || L33_.size() == 0) {
        throw std::runtime_error("Cholesky decomposition not computed.");
    }

    if (b.size() != 3 * block_size_) {
        throw std::invalid_argument("Right-hand side vector has incorrect size.");
    }

    // Partition the vector b into b1, b2, b3
    Eigen::VectorXd b1 = b.segment(0, block_size_);
    Eigen::VectorXd b2 = b.segment(block_size_, block_size_);
    Eigen::VectorXd b3 = b.segment(2 * block_size_, block_size_);

    // Forward Substitution: Solve L_post * y = b
    // y1 = L11^{-1} * b1
    Eigen::VectorXd y1 = L11_.triangularView<Eigen::Lower>().solve(b1);

    // y2 = L22^{-1} * (b2 - L21 * y1)
    Eigen::VectorXd y2_intermediate = b2 - L21_ * y1;
    Eigen::VectorXd y2 = L22_.triangularView<Eigen::Lower>().solve(y2_intermediate);

    // y3 = L33^{-1} * (b3 - L31 * y1 - L32 * y2)
    Eigen::VectorXd y3_intermediate = b3 - L31_ * y1 - L32_ * y2;
    Eigen::VectorXd y3 = L33_.triangularView<Eigen::Lower>().solve(y3_intermediate);

    // Combine y1, y2, y3
    Eigen::VectorXd y(3 * block_size_);
    y << y1, y2, y3;

    return y;
}

Eigen::VectorXd BlockCholesky::backwardSolve(const Eigen::VectorXd& y) const {
    if (L11_.size() == 0 || L22_.size() == 0 || L33_.size() == 0) {
        throw std::runtime_error("Cholesky decomposition not computed.");
    }

    if (y.size() != 3 * block_size_) {
        throw std::invalid_argument("Intermediate vector y has incorrect size.");
    }

    // Partition the vector y into y1, y2, y3
    Eigen::VectorXd y1 = y.segment(0, block_size_);
    Eigen::VectorXd y2 = y.segment(block_size_, block_size_);
    Eigen::VectorXd y3 = y.segment(2 * block_size_, block_size_);

    // Backward Substitution: Solve L_post^T * x = y
    // x3 = L33^{-T} * y3
    Eigen::VectorXd x3 = L33_.transpose().triangularView<Eigen::Upper>().solve(y3);

    // x2 = L22^{-T} * (y2 - L32^T * x3)
    Eigen::VectorXd x2_intermediate = y2 - L32_.transpose() * x3;
    Eigen::VectorXd x2 = L22_.transpose().triangularView<Eigen::Upper>().solve(x2_intermediate);

    // x1 = L11^{-T} * (y1 - L21^T * x2 - L31^T * x3)
    Eigen::VectorXd x1_intermediate = y1 - L21_.transpose() * x2 - L31_.transpose() * x3;
    Eigen::VectorXd x1 = L11_.transpose().triangularView<Eigen::Upper>().solve(x1_intermediate);

    // Combine x1, x2, x3
    Eigen::VectorXd x(3 * block_size_);
    x << x1, x2, x3;

    return x;
}

Eigen::VectorXd BlockCholesky::solve(const Eigen::VectorXd& b) const {
    Eigen::VectorXd y = forwardSolve(b);
    Eigen::VectorXd x = backwardSolve(y);
    return x;
}

double BlockCholesky::logDeterminant() const {
    return log_det_;
}

} // namespace block_cholesky
