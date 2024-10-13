#ifndef BLOCK_CHOLESKY_H
#define BLOCK_CHOLESKY_H

#include <Eigen/Dense>
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
    void setPrecisionBlocks(const Eigen::MatrixXd& Q_pp, 
                            const Eigen::MatrixXd& Q_tt, 
                            const Eigen::MatrixXd& Q_ff,
                            const Eigen::MatrixXd& Q_pt, 
                            const Eigen::MatrixXd& Q_pf, 
                            const Eigen::MatrixXd& Q_tf);

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
    Eigen::MatrixXd Q_pp_;
    Eigen::MatrixXd Q_tt_;
    Eigen::MatrixXd Q_ff_;
    Eigen::MatrixXd Q_pt_;
    Eigen::MatrixXd Q_pf_;
    Eigen::MatrixXd Q_tf_;

    // Cholesky factors
    Eigen::MatrixXd L11_;
    Eigen::MatrixXd L21_;
    Eigen::MatrixXd L31_;
    Eigen::MatrixXd L22_;
    Eigen::MatrixXd L32_;
    Eigen::MatrixXd L33_;

    // Log determinant
    double log_det_;

    /**
     * @brief Helper function to perform Cholesky decomposition on a single block.
     *
     * @param block The matrix block to decompose.
     * @return The lower triangular matrix from Cholesky decomposition, or an empty matrix if decomposition fails.
     */
    Eigen::MatrixXd choleskyDecompose(const Eigen::MatrixXd& block) const;
};

} // namespace block_cholesky

#endif // BLOCK_CHOLESKY_H
