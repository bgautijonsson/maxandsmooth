#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <RcppEigen.h>
#include <nlopt.hpp>
#include <Eigen/Sparse>
#include <omp.h>


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(nloptr)]]
using namespace autodiff;
using namespace Eigen;

#pragma omp declare reduction(+: autodiff::var: omp_out += omp_in)

/**
 * @brief Computes the simultaneous log-likelihood of the GEV distribution for all locations.
 */
var loglik(const ArrayXvar& x, const Eigen::MatrixXd& data) {
    int N = data.rows();
    int P = data.cols();
    var loglik = 0;

    #pragma omp parallel for reduction(+:loglik)
    for (int p = 0; p < P; ++p) {
        // Extract parameters for location p
        var mu = exp(x[p]);
        var sigma = exp(x[p] + x[P + p]);
        var xi = pow(1 + exp(-x[2 * P + p]), -1);

        // Vectorize z calculation
        ArrayXvar z = (data.col(p).array() - mu) / sigma;

        // Vectorize likelihood calculation
        if (xi < 1e-6) {
            loglik -= (log(sigma) * N + z.sum() + exp(-z).sum());
        } else {
            ArrayXvar t = 1 + xi * z;
            loglik -= (log(sigma) * N + (1.0 + 1.0/xi) * log(t).sum() + 
                      exp(-1.0/xi * log(t)).sum());
        }

        // Priors remain the same
        loglik -= 0.5 * pow(x[p], 2);
        loglik -= 0.5 * pow(x[P + p], 2);
        loglik -= 0.5 * pow(x[2 * P + p], 2);
    }

    return loglik;
}


/**
 * @brief Wrapper function for NLopt optimization that computes objective value and gradient
 */
double objective(unsigned n, const double* x, double* grad, void* f_data) {
    const Eigen::MatrixXd* data = reinterpret_cast<const Eigen::MatrixXd*>(f_data);
    
    // Convert x to ArrayXvar for autodiff
    ArrayXvar params(n);
    for(unsigned i = 0; i < n; ++i) {
        params[i] = x[i];
    }
    
    // Compute objective and gradient if needed
    var obj = -loglik(params, *data);  // Negative because we're minimizing
    
    if(grad) {
        ArrayXd g = gradient(obj, params);
        for(unsigned i = 0; i < n; ++i) {
            grad[i] = g[i];
        }
    }
    
    return val(obj);
}

// [[Rcpp::export]]
Rcpp::List fit_gev(Eigen::MatrixXd& data) {
    int P = data.cols();
    int n_params = 3 * P;  // 3 parameters per location
    
    // Initialize optimizer
    nlopt::opt opt(nlopt::LD_LBFGS, n_params);
    
    // Set initial values
    std::vector<double> x(n_params, 0.0);
    
    // Set optimization parameters
    opt.set_min_objective(objective, &data);
    opt.set_ftol_rel(1e-8);
    opt.set_maxeval(1000);
    opt.set_xtol_rel(1e-6);
    opt.set_initial_step(1.0);
    
    // Run optimization
    double min_obj;
    try {
        nlopt::result result = opt.optimize(x, min_obj);
    } catch(std::exception &e) {
        Rcpp::stop("Optimization failed: " + std::string(e.what()));
    }
    
    // Convert optimal parameters to ArrayXvar for final gradient and Hessian
    ArrayXvar params_var(n_params);
    for(int i = 0; i < n_params; ++i) {
        params_var[i] = x[i];
    }
    
    // Compute final gradient and objective
    var obj = loglik(params_var, data);
    VectorXd grad;
    MatrixXd dense_hess = hessian(obj, params_var, grad);
    
    // Convert to sparse matrix
    int n = dense_hess.rows();
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(n * 3);  // Reserve space for block diagonal structure
    
    // Add non-zero elements to triplet list
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(std::abs(dense_hess(i,j)) > 1e-10) {  // Threshold for numerical zeros
                tripletList.push_back(T(i, j, dense_hess(i,j)));
            }
        }
    }
    
    // Create sparse matrix
    SparseMatrix<double> sparse_hess(n, n);
    sparse_hess.setFromTriplets(tripletList.begin(), tripletList.end());

    // Calculate sparse Cholesky decomposition of negative Hessian (precision matrix)
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> llt;
    SparseMatrix<double> precision = -sparse_hess;  // Negative Hessian is precision matrix
    llt.compute(precision);
    
    // Check if decomposition was successful
    if (llt.info() != Eigen::Success) {
        Rcpp::warning("Cholesky decomposition failed!");
    }
    
    // Get the sparse Cholesky factor
    SparseMatrix<double> L = llt.matrixL();
    
    // Return results
    return Rcpp::List::create(
        Rcpp::Named("parameters") = x,
        Rcpp::Named("L") = Rcpp::wrap(L)
    );
}

