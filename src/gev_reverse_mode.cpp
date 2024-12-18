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

VectorXd global_index;
VectorXd global_n_values;
VectorXd global_values;
double global_log_det;

/**
 * @brief Computes the log probability density function for a GEV distribution
 */
var gev_lpdf(const ArrayXd& y, const var& mu, const var& sigma, const var& xi) {
    ArrayXvar z = (y - mu) / sigma;
    int N = y.size();
    var ll;
    
    if (xi < 1e-6) {
        ll = -(log(sigma) * N + z.sum() + exp(-z).sum());
    } else {
        ArrayXvar t = 1 + xi * z;
        ll = -(log(sigma) * N + (1.0 + 1.0/xi) * log(t).sum() + 
               exp(-1.0/xi * log(t)).sum());
    }
    return ll;
}

/**
 * @brief Computes the cumulative distribution function for a GEV distribution
 */
ArrayXvar gev_cdf(const ArrayXd& y, const var& mu, const var& sigma, const var& xi) {
    ArrayXvar z = (y - mu) / sigma;
    ArrayXvar cdf(z.size());
    
    if (xi < 1e-6) {
        // Gumbel case (xi ≈ 0)
        cdf = exp(-exp(-z));
    } else {
        // General case
        ArrayXvar t = 1 + xi * z;
        cdf = exp(-pow(t, -1/xi));
    }
    return cdf;
}

/**
 * @brief Approximates the inverse normal CDF (probit function) using Acklam's algorithm
 * Modified to work with autodiff::var
 */
var normal_quantile(const var& p) {
    // Coefficients for Acklam's approximation
    const double a1 = -3.969683028665376e+01;
    const double a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 = 2.506628277459239e+00;
    
    const double b1 = -5.447609879822406e+01;
    const double b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;
    
    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00;
    const double c6 = 2.938163982698783e+00;
    
    const double d1 = 7.784695709041462e-03;
    const double d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;
    
    // Break points
    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;
    
    var x;
    
    if (p < p_low) {
        // Lower region
        var q = sqrt(-2.0 * log(p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    } else if (p <= p_high) {
        // Central region
        var q = p - 0.5;
        var r = q * q;
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        // Upper region
        var q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    
    return x;
}

/**
 * @brief Transforms GEV margins to standard normal using probability integral transform
 */
ArrayXvar gev_to_normal(const ArrayXd& y, const var& mu, const var& sigma, const var& xi) {
    // First get uniform margins using GEV CDF
    ArrayXvar u = gev_cdf(y, mu, sigma, xi);
    
    // Transform to normal using inverse normal CDF (probit function)
    int n = u.size();
    ArrayXvar z(n);
    for(int i = 0; i < n; i++) {
        z(i) = normal_quantile(u(i));
    }
    
    return z;
}

/**
 * @brief Computes the log-likelihood of the multivariate normal distribution 
 * for the cholesky factor of a constant, sparse precision matrix Q
 */
var mvnormal_prec_chol_lpdf(
    const MatrixXvar& x
) {
    int N = x.rows();
    var ll = 0;
    var quadform = 0;
    
    // Initialize q vector for quadratic form calculation
    ArrayXvar q = ArrayXvar::Zero(N);
    
    // Calculate quadratic form (y-x)'Q(y-x) using sparse representation
    int counter = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < global_n_values[i]; j++) {
            int idx = global_index[counter];  
            q[i] += global_values[counter] * (x(i) - x(idx));
            counter++;
        }
        // Accumulate the quadratic form
        quadform += q[i] * q[i];
    }
    
    // Compute log-likelihood: log|Q|/2 - 1/2(y-x)'Q(y-x)
    ll = global_log_det - 0.5 * quadform;
    
    return ll;
}

/**
 * @brief Computes the log-likelihood of a Gaussian copula
 */
var gaussian_copula_lpdf(
    const ArrayXvar& x,
    const MatrixXd& data
) {
    int N = data.rows();
    int P = data.cols();
    
    // Transform margins to standard normal
    MatrixXvar Z(N, P);
    for (int p = 0; p < P; ++p) {
        var mu = exp(x[p]);
        var sigma = exp(x[p] + x[P + p]);
        var xi = pow(1 + exp(-x[2 * P + p]), -1);
        
        Z.col(p) = gev_to_normal(data.col(p).array(), mu, sigma, xi);
    }

    var copula_ll = 0;
    for (int i = 0; i < N; i++) {
        copula_ll += mvnormal_prec_chol_lpdf(Z.row(i));
        copula_ll -= 0.5 * Z.row(i).array().square().sum();
    }
    
    //copula_ll += mvnormal_prec_chol_lpdf(Z.row(1)) - 0.5 * Z.row(1).array().square().sum();
    return copula_ll;
}


/**
 * @brief Computes the simultaneous log-likelihood of the GEV distribution for all locations.
 */
var marginal_loglik(const ArrayXvar& x, const Eigen::MatrixXd& data) {
    int N = data.rows();
    int P = data.cols();
    var loglik = 0;

    for (int p = 0; p < P; ++p) {
        // Extract parameters for location p
        var mu = exp(x[p]);
        var sigma = exp(x[p] + x[P + p]);
        var xi = pow(1 + exp(-x[2 * P + p]), -1);

        // Use the new gev_lpdf function
        loglik += gev_lpdf(data.col(p).array(), mu, sigma, xi);

        // Priors
        loglik -= 0.5 * pow(x[p], 2);
        loglik -= 0.5 * pow(x[P + p], 2);
        loglik -= 0.5 * pow(x[2 * P + p], 2);
    }

    return loglik;
}

/**
 * @brief Combined log-likelihood for GEV margins and Gaussian copula
 */
var loglik(
    const ArrayXvar& x, 
    const MatrixXd& data
) {    
    // Add GEV marginal log-likelihoods
    var ll = marginal_loglik(x, data);
    
    // Add Gaussian copula log-likelihood
    ll += gaussian_copula_lpdf(x, data);
    
    return ll;
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
Rcpp::List fit_gev(
    Eigen::MatrixXd& data,
    Eigen::VectorXd& index,
    Eigen::VectorXd& n_values,
    Eigen::VectorXd& values,
    double log_det
) {
    int P = data.cols();
    int n_params = 3 * P;  // 3 parameters per location

    global_index = index;
    global_n_values = n_values;
    global_values = values;
    global_log_det = log_det;
    
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
    
    // Add non-zero elements to triplet list
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(std::abs(dense_hess(i,j)) > 1e-14) {  // Threshold for numerical zeros
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
        Rcpp::Named("L") = Rcpp::wrap(L),
        Rcpp::Named("Hessian") = Rcpp::wrap(sparse_hess)
    );
}

