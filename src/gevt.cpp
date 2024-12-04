#include "gevt.h"

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


namespace gevt {

/**
 * @brief Computes the log-likelihood of the GEV distribution.
 *
 * @param params Transformed GEV parameters (psi, tau, phi).
 * @param data Observed data points.
 * @return The computed log-likelihood.
 */
dual loglik(dual psi, dual tau, dual phi, dual gamma, const VectorXd& data) {
    dual mu0 = exp(psi);
    dual sigma = exp(psi + tau);
    dual xi = pow(1 + exp(-phi), -1);
    dual delta = 0.05 * pow(1 + exp(-gamma), -1);

    int n = data.size();
    dual loglik = 0;

    for(int i = 0; i < n; ++i) {
      dual mu = mu0 * (1 + delta * i);
      dual z = (data(i) - mu) / sigma;

      if (xi < 1e-6) {
          loglik -= log(sigma) + z + exp(-z);
      } else {
          dual t = 1 + xi * z;
          loglik -= log(sigma) + (1.0 + 1.0 / xi) * log(t) + pow(t, -1.0 / xi);
      }
    }

    // Priors
    loglik -= 0.5 * pow(psi, 2);
    loglik -= 0.5 * pow(tau, 2);
    loglik -= 0.5 * pow(phi, 2);
    loglik -= 0.5 * pow(gamma, 2);

    return loglik;
}

/**
 * @brief Evaluates the log-likelihood value for given GEV parameters and data.
 *
 * @param params The transformed GEV parameters (psi, tau, phi).
 * @param data The observed data points.
 * @return The log-likelihood value as a double.
 */
double loglik_value(dual psi, dual tau, dual phi, dual delta,   const Eigen::VectorXd& data) {
    auto f = [&data](dual psi, dual tau, dual phi, dual delta) { return loglik(psi, tau, phi, delta, data); };
    return val(f(psi, tau, phi, delta));
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
    dual psi = x[0];
    dual tau = x[1];
    dual phi = x[2];
    dual delta = x[3];

    double loglik_val = loglik_value(psi, tau, phi, delta, dataVec);

    if (grad) {
        VectorXdual grads;
        auto f = [&dataVec](dual psi, dual tau, dual phi, dual delta) -> dual { 
            return loglik(psi, tau, phi, delta, dataVec); 
        };
        grads = gradient(f, wrt(psi, tau, phi, delta), at(psi, tau, phi, delta));
        for (int i = 0; i < 4; ++i) {
            grad[i] = -val(grads(i));
        }
    }

    return -loglik_val;  // Negative because we're minimizing
}

/**
 * @brief Performs maximum likelihood estimation for GEV parameters.
 *
 * @param data The observed data points.
 * @return The estimated GEV parameters as Eigen::Vector3d.
 */
Eigen::Vector4d mle(const Eigen::VectorXd& data) {
    nlopt::opt opt(nlopt::LD_LBFGS, 4);  // 4 parameters
    Eigen::Vector4d initial_params(std::log(5.0), std::log(5.0) - std::log(1.0), std::log(0.1) - std::log(0.9), std::log(0.05) - std::log(0.95));

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

    return Eigen::Vector4d(x[0], x[1], x[2], x[3]);
}

/**
 * @brief Alternate version of the log-likelihood where the parameters are twice differentiable.
 *
 * @param params The transformed GEV parameters (psi, tau, phi).
 * @param data The observed data points.
 * @return The computed log-likelihood.
 */
dual2nd loglik_2nd(dual2nd psi, dual2nd tau, dual2nd phi, dual2nd gamma, const VectorXd& data) {
    dual2nd mu0 = exp(psi);
    dual2nd sigma = exp(psi + tau);
    dual2nd xi = pow(1 + exp(-phi), -1);
    dual2nd delta = 0.05 * pow(1 + exp(-gamma), -1);

    int n = data.size();
    dual2nd loglik = 0;

    for(int i = 0; i < n; ++i) {
        dual2nd mu = mu0 * (1 + delta * i);
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
 * @brief Computes the Hessian of the log-likelihood of the GEV distribution.
 *
 * @param params The transformed GEV parameters (psi, tau, phi).
 * @param data The observed data points.
 * @return The computed Hessian matrix.
 */
Eigen::MatrixXd loglik_hessian(dual2nd psi, dual2nd tau, dual2nd phi, dual2nd delta, const Eigen::VectorXd& data) {

    // Define a lambda function that takes dual parameters and returns the log-likelihood
    auto f = [&data](dual2nd psi, dual2nd tau, dual2nd phi, dual2nd delta) -> dual2nd {
        return loglik_2nd(psi, tau, phi, delta, data);
    };

    // Compute the Hessian matrix using autodiff's hessian function
    Eigen::Matrix<dual2nd, Eigen::Dynamic, Eigen::Dynamic> hess_dual = autodiff::hessian(f, wrt(psi, tau, phi, delta), at(psi, tau, phi, delta));

    // Convert the Hessian from dual to double precision
    Eigen::MatrixXd hess = hess_dual.cast<double>();

    return hess;
}

/**
 * @brief Performs maximum likelihood estimation for GEV parameters for multiple locations in parallel.
 *
 * @param data A matrix where each column represents data for a location.
 * @return A list containing the MLEs and Hessians for each location.
 */
Rcpp::List mle_multiple(Eigen::MatrixXd& data) {
    int n_locations = data.cols();
    Eigen::MatrixXd results(n_locations, 4);
    Eigen::MatrixXd hessians(n_locations, 16);
    
    #pragma omp parallel for
    for (int i = 0; i < n_locations; ++i) {
        Eigen::Vector4d mle_result = mle(data.col(i));
        dual2nd psi = mle_result(0);
        dual2nd tau = mle_result(1);
        dual2nd phi = mle_result(2);
        dual2nd delta = mle_result(3);
        Eigen::MatrixXd hess = loglik_hessian(psi, tau, phi, delta, data.col(i));
        results.row(i) = mle_result;
        hessians.row(i) = Eigen::Map<const Eigen::VectorXd>(hess.data(), hess.size());
    }

    return Rcpp::List::create(
        Rcpp::Named("mles") = results,
        Rcpp::Named("hessians") = hessians
    );
}

} // namespace gevt