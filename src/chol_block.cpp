#include <RcppEigen.h>
#include <stdexcept>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]


// [[Rcpp::export]]
Rcpp::List chol_block(
    const Eigen::VectorXd& Q_pp_vec,
    const Eigen::VectorXd& Q_tt_vec,
    const Eigen::VectorXd& Q_ff_vec,
    const Eigen::VectorXd& Q_pt_vec,
    const Eigen::VectorXd& Q_pf_vec,
    const Eigen::VectorXd& Q_tf_vec
) {

    Eigen::SparseMatrix<double> Q_pp = Q_pp_vec.asDiagonal().toDenseMatrix().sparseView();
    Eigen::SparseMatrix<double> Q_tt = Q_tt_vec.asDiagonal().toDenseMatrix().sparseView();
    Eigen::SparseMatrix<double> Q_ff = Q_ff_vec.asDiagonal().toDenseMatrix().sparseView();
    Eigen::SparseMatrix<double> Q_pt = Q_pt_vec.asDiagonal().toDenseMatrix().sparseView();
    Eigen::SparseMatrix<double> Q_pf = Q_pf_vec.asDiagonal().toDenseMatrix().sparseView();
    Eigen::SparseMatrix<double> Q_tf = Q_tf_vec.asDiagonal().toDenseMatrix().sparseView();

    int block_size = Q_pp.rows();

    // Step 1: Compute L11 = Cholesky(Q_pp + tau_psi Q_prior)
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> llt1(Q_pp);
    Eigen::SparseMatrix<double> L11 = llt1.matrixL();
    if (L11.size() == 0) {
        return false; // Decomposition failed
    }

    // Step 2: Compute L21 = Q_pt * L11^{-T}
    Eigen::SparseMatrix<double> L11_invT = L11.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(block_size, block_size)).sparseView();
    Eigen::SparseMatrix<double> L21 = Q_pt * L11_invT;

    // Step 3: Compute L31 = Q_pf * L11^{-T}
    Eigen::SparseMatrix<double> L31 = Q_pf * L11_invT;

    // Step 4: Compute S22 = Q_tt + tau_tau Q_prior - L21 * L21^T
    Eigen::SparseMatrix<double> S22 = Q_tt - L21 * L21.transpose();
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> llt2(S22);
    Eigen::SparseMatrix<double> L22 = llt2.matrixL();
    if (L22.size() == 0) {
        return false; // Decomposition failed
    }

    // Step 5: Compute L32 = (Q_tf - L31 * L21^T) * L22^{-T}
    Eigen::SparseMatrix<double> L22_invT = L22.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(block_size, block_size)).sparseView();
    Eigen::SparseMatrix<double> L32 = (Q_tf - L31 * L21.transpose()) * L22_invT;

    // Step 6: Compute S33 = Q_ff + tau_phi Q_prior - L31 * L31^T - L32 * L32^T
    Eigen::SparseMatrix<double> S33 = Q_ff - L31 * L31.transpose() - L32 * L32.transpose();
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> llt3(S33);
    Eigen::SparseMatrix<double> L33 = llt3.matrixL();
    if (L33.size() == 0) {
        return false; // Decomposition failed
    }

    return Rcpp::List::create(
        Rcpp::Named("L11") = L11,
        Rcpp::Named("L21") = L21,
        Rcpp::Named("L31") = L31,
        Rcpp::Named("L22") = L22,
        Rcpp::Named("L32") = L32,
        Rcpp::Named("L33") = L33
    );
}
