#ifndef BLOCK_CHOLESKY_H
#define BLOCK_CHOLESKY_H

#include <Eigen/Sparse>
#include <vector>

namespace block_cholesky {

/**
 * @brief Class to perform Block-Cholesky decomposition on a precision matrix.
 */
class BlockCholesky {
public:
    /**
     * @brief Constructs the BlockCholesky object.
     *
     * @param n_blocks The number of parameter blocks (e.g., 3 for psi, tau, phi).
     * @param block_size The number of spatial locations per block.
     */
    BlockCholesky(int n_blocks, int block_size);

    /**
     * @brief Sets the precision matrix blocks.
     *
     * @param Q_pp \( Q_{\psi\psi} + \tau_\psi Q_{\text{prior}} \)
     * @param Q_tt \( Q_{\tau\tau} + \tau_\tau Q_{\text{prior}} \)
     * @param Q_ff \( Q_{\phi\phi} + \tau_\phi Q_{\text{prior}} \)
     * @param Q_pt \( Q_{\psi\tau} \)
     * @param Q_pf \( Q_{\psi\phi} \)
     * @param Q_tf \( Q_{\tau\phi} \)
     */
    void setPrecisionBlocks(
        const Eigen::SparseMatrix<double>& Q_pp, 
        const Eigen::SparseMatrix<double>& Q_tt, 
        const Eigen::SparseMatrix<double>& Q_ff,
        const Eigen::SparseMatrix<double>& Q_pt, 
        const Eigen::SparseMatrix<double>& Q_pf, 
        const Eigen::SparseMatrix<double>& Q_tf,
        const Eigen::SparseMatrix<double>& Q_prior
    );

    /**
     * @brief Sets the tau vector.
     *
     * @param tau The tau vector.
     */
    void setTau(const Eigen::VectorXd& tau);

    /**
     * @brief Performs the Block-Cholesky decomposition as per the algorithm.
     *
     * @return True if decomposition is successful, false otherwise.
     */
    bool computeDecomposition();

    /**
     * @brief Performs the forward substitution step \( L_{\text{post}} y = b \).
     *
     * @param b The right-hand side vector.
     * @return The intermediate vector y.
     */
    Eigen::VectorXd forwardSolve(const Eigen::VectorXd& b) const;

    /**
     * @brief Performs the backward substitution step \( L_{\text{post}}^T x = y \).
     *
     * @param y The intermediate vector from forward substitution.
     * @return The solution vector x.
     */
    Eigen::VectorXd backwardSolve(const Eigen::VectorXd& y) const;

    /**
     * @brief Solves the linear system \( Q_{\text{post}} x = b \) using both forward and backward passes.
     *
     * @param b The right-hand side vector.
     * @return The solution vector x.
     */
    Eigen::VectorXd solve(const Eigen::VectorXd& b) const;

    /**
     * @brief Retrieves the log determinant of the precision matrix.
     *
     * @return The log determinant value.
     */
    double logDeterminant() const;

private:
    int n_blocks_;
    int block_size_;

    // Precision matrix components
    Eigen::SparseMatrix<double> Q_pp_;
    Eigen::SparseMatrix<double> Q_tt_;
    Eigen::SparseMatrix<double> Q_ff_;
    Eigen::SparseMatrix<double> Q_pt_;
    Eigen::SparseMatrix<double> Q_pf_;
    Eigen::SparseMatrix<double> Q_tf_;
    Eigen::SparseMatrix<double> Q_prior_;
    Eigen::VectorXd tau_;
    // Cholesky factors
    Eigen::SparseMatrix<double> L11_;
    Eigen::SparseMatrix<double> L21_;
    Eigen::SparseMatrix<double> L31_;
    Eigen::SparseMatrix<double> L22_;
    Eigen::SparseMatrix<double> L32_;
    Eigen::SparseMatrix<double> L33_;

    // Log determinant
    double log_det_;

    /**
     * @brief Helper function to perform Cholesky decomposition on a single block.
     *
     * @param block The matrix block to decompose.
     * @return The lower triangular matrix from Cholesky decomposition, or an empty matrix if decomposition fails.
     */
    Eigen::SparseMatrix<double> choleskyDecompose(const Eigen::SparseMatrix<double>& block);
};

} // namespace block_cholesky

#endif // BLOCK_CHOLESKY_H
