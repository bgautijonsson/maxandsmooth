#include <RcppEigen.h>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <nlopt.hpp>
#include <omp.h>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(autodiff)]]
// [[Rcpp::depends(nloptr)]]

using namespace Rcpp;
using namespace autodiff;
using namespace Eigen;

/**
 * @brief Computes the log-likelihood of the GEV distribution.
 *
 * @param params Transformed GEV parameters (psi, tau, phi).
 * @param data Observed data points.
 * @return The computed log-likelihood.
 */
dual2nd gev_loglik(dual2nd psi, dual2nd tau, dual2nd phi, const VectorXd& data) {

    dual2nd mu = exp(psi);
    dual2nd sigma = exp(psi + tau);
    dual2nd xi = pow(1 + exp(-phi), -1);

    int n = data.size();
    dual2nd loglik = 0;

    for(int i = 0; i < n; ++i) {
        dual2nd z = (data(i) - mu) / sigma;

        if (xi < 1e-6) {
            loglik -= log(sigma) + z + exp(-z);
        } else {
            dual2nd t = 1 + xi * z;
            loglik -= log(sigma) + (1.0 + 1.0 / xi) * log(t) + pow(t, -1.0 / xi);
        }
    }

    return loglik;
}

/**
 * @brief Calculates the gradient of the GEV log-likelihood.
 *
 * @param params The GEV parameters (mu, sigma, xi).
 * @param data The observed data points.
 * @return The gradient vector as Eigen::VectorXd.
 */
Eigen::VectorXd gev_gradient(dual2nd psi, dual2nd tau, dual2nd phi, const Eigen::VectorXd& data) {
    VectorXdual grad;

    auto f = [&data](dual2nd psi, dual2nd tau, dual2nd phi) { return gev_loglik(psi, tau, phi, data); };
    grad = gradient(f, wrt(psi, tau, phi), at(psi, tau, phi));

    return grad.cast<double>();
}

/**
 * @brief Evaluates the log-likelihood value for given GEV parameters and data.
 *
 * @param params The GEV parameters (mu, sigma, xi).
 * @param data The observed data points.
 * @return The log-likelihood value as a double.
 */
double gev_loglik_value(dual2nd psi, dual2nd tau, dual2nd phi, const Eigen::VectorXd& data) {
    auto f = [&data](dual2nd psi, dual2nd tau, dual2nd phi) { return gev_loglik(psi, tau, phi, data); };
    return val(f(psi, tau, phi));
}

/**
 * @brief Objective function for NLopt optimization.
 *
 * @param n Number of parameters.
 * @param x Pointer to the parameter array.
 * @param grad Pointer to the gradient array.
 * @param f_data Pointer to additional data (Eigen::VectorXd).
 * @return The objective function value as a double.
 */
double objective(unsigned n, const double* x, double* grad, void* f_data) {
    const Eigen::VectorXd* pData = reinterpret_cast<const Eigen::VectorXd*>(f_data);
    const Eigen::VectorXd& dataVec = *pData;
    dual2nd psi = x[0];
    dual2nd tau = x[1];
    dual2nd phi = x[2];

    double loglik = gev_loglik_value(psi, tau, phi, dataVec);

    if (grad) {
        Eigen::Vector3d gradient = -gev_gradient(psi, tau, phi, dataVec);
        for (int i = 0; i < 3; ++i) {
            grad[i] = gradient(i);
        }
    }

    return -loglik;  // Negative because we're minimizing
}




/**
 * @brief Performs maximum likelihood estimation for GEV parameters.
 *
 * @param data The observed data points.
 * @return The estimated GEV parameters as Eigen::Vector3d.
 */
Eigen::Vector3d gev_mle(const Eigen::VectorXd& data) {
    nlopt::opt opt(nlopt::LD_LBFGS, 3);  // 3 parameters
    Eigen::Vector3d initial_params(std::log(5.0), std::log(5.0) - std::log(1.0), std::log(0.1) - std::log(0.9));

    opt.set_min_objective(objective, (void*)&data);
    opt.set_ftol_rel(1e-6);
    opt.set_xtol_rel(1e-6); 
    opt.set_maxeval(1000);

    std::vector<double> x(initial_params.data(), initial_params.data() + initial_params.size());
    double minf;

    try {
        nlopt::result result = opt.optimize(x, minf);
    }
    catch(std::exception &e) {
        Rcpp::stop("NLopt failed: " + std::string(e.what()));
    }

    return Eigen::Vector3d(x[0], x[1], x[2]);
}


/**
 * @brief Calculates the Hessian matrix of the GEV log-likelihood.
 *
 * This function computes the second-order derivatives (Hessian) of the
 * GEV log-likelihood with respect to the parameters (mu, sigma, xi).
 *
 * @param psi Transformed parameter psi = log(mu)
 * @param tau Transformed parameter tau = log(sigma) - psi
 * @param phi Transformed parameter phi = logit(xi)
 * @param data The observed data points as an Eigen::VectorXd.
 * @return The Hessian matrix as an Eigen::MatrixXd.
 */
Eigen::MatrixXd gev_hessian(dual2nd psi, dual2nd tau, dual2nd phi, const Eigen::VectorXd& data) {

    // Define a lambda function that takes dual parameters and returns the log-likelihood
    auto f = [&data](dual2nd psi, dual2nd tau, dual2nd phi) -> dual2nd {
        return gev_loglik(psi, tau, phi, data);
    };

    // Compute the Hessian matrix using autodiff's hessian function
    Eigen::Matrix<dual2nd, Eigen::Dynamic, Eigen::Dynamic> hess_dual = autodiff::hessian(f, wrt(psi, tau, phi), at(psi, tau, phi));

    // Convert the Hessian from dual to double precision
    Eigen::MatrixXd hess = hess_dual.cast<double>();

    return hess;
}

/**
 * @brief Computes Maximum Likelihood Estimates for multiple locations in parallel.
 *
 * @param data A matrix where each column represents data for a location.
 * @return A matrix of MLEs with each row corresponding to a location's parameters.
 */
// [[Rcpp::export]]
Rcpp::List gev_mle_multiple(Eigen::MatrixXd& data) {
    int n_locations = data.cols();
    Eigen::MatrixXd results(n_locations, 3);
    Eigen::MatrixXd hessians(n_locations, 9);
    
    #pragma omp parallel for
    for (int i = 0; i < n_locations; ++i) {
        Eigen::Vector3d mle_result = gev_mle(data.col(i));
        dual2nd psi = mle_result(0);
        dual2nd tau = mle_result(1);
        dual2nd phi = mle_result(2);
        Eigen::MatrixXd hess = gev_hessian(psi, tau, phi, data.col(i));
        results.row(i) = mle_result;
        hessians.row(i) = hess.reshaped();
    }

    return Rcpp::List::create(
        Rcpp::Named("mles") = results,
        Rcpp::Named("hessians") = hessians
    );
}
