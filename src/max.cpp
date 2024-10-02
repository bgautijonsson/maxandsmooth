#include <RcppEigen.h>
#include <cmath>
#include <boost/math/tools/minima.hpp>
#include <omp.h>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace Eigen;

// Log-likelihood function for the GEV distribution
double gev_log_likelihood(const Eigen::VectorXd& data, double mu, double sigma, double xi) {
    int n = data.size();
    double log_lik = 0.0;

    for (int i = 0; i < n; ++i) {
        double z = (data[i] - mu) / sigma;
        if (xi != 0) {
            double t = 1 + xi * z;
            if (t <= 0) return -INFINITY; // Invalid parameter region
            log_lik -= std::log(sigma) + (1 + 1/xi) * std::log(t) + std::pow(t, -1/xi);
        } else {
            log_lik -= std::log(sigma) + z + std::exp(-z);
        }
    }

    return log_lik;
}

// Gradient of the log-likelihood function
Eigen::VectorXd gev_log_likelihood_gradient(const Eigen::VectorXd& data, double mu, double sigma, double xi) {
    int n = data.size();
    Eigen::VectorXd grad(3);
    grad.setZero();

    for (int i = 0; i < n; ++i) {
        double z = (data[i] - mu) / sigma;
        if (xi != 0) {
            double t = 1 + xi * z;
            if (t <= 0) return Eigen::VectorXd::Constant(3, -INFINITY); // Invalid parameter region
            double t_inv = std::pow(t, -1/xi);
            grad[0] += (1 + 1/xi) * (1 / t) - t_inv / sigma;
            grad[1] += (1 + 1/xi) * (z / t) - t_inv * z / sigma;
            grad[2] += (1/xi) * (1/xi + 1) * (z / t) - t_inv * z * z / sigma;
        } else {
            grad[0] += 1 - std::exp(-z);
            grad[1] += z - std::exp(-z) * z;
            grad[2] += 0; // No contribution from xi when xi == 0
        }
    }

    return grad;
}

// Function to minimize (negative log-likelihood)
struct GEVNegLogLikelihood {
    const Eigen::VectorXd& data;
    double sigma;
    double xi;

    GEVNegLogLikelihood(const Eigen::VectorXd& data, double sigma, double xi) 
        : data(data), sigma(sigma), xi(xi) {}

    double operator()(double mu) const {
        return -gev_log_likelihood(data, mu, sigma, xi);
    }
};

// Updated function to perform MLE for the GEV distribution using Brent's method
// Now returns a vector instead of a list
// [[Rcpp::export]]
Eigen::Vector3d gev_mle(const Eigen::VectorXd& data) {
    // Initial guesses
    double mu = 5;
    double sigma = 1;
    double xi = 0.01;

    // Optimization settings
    int max_iter = 100;
    double tol = 1e-6;

    for (int i = 0; i < max_iter; ++i) {
        // Optimize mu
        GEVNegLogLikelihood f_mu(data, sigma, xi);
        std::pair<double, double> r_mu = boost::math::tools::brent_find_minima(
            f_mu, mu - sigma, mu + sigma, 20);
        mu = r_mu.first;

        // Optimize sigma (ensure it's positive)
        auto f_sigma = [&](double s) { 
            return -gev_log_likelihood(data, mu, std::max(s, 1e-6), xi); 
        };
        std::pair<double, double> r_sigma = boost::math::tools::brent_find_minima(
            f_sigma, std::max(0.1 * sigma, 1e-6), 10 * sigma, 20);
        sigma = std::max(r_sigma.first, 1e-6);

        // Optimize xi
        auto f_xi = [&](double x) { 
            return -gev_log_likelihood(data, mu, sigma, x); 
        };
        std::pair<double, double> r_xi = boost::math::tools::brent_find_minima(
            f_xi, xi - 1, xi + 1, 20);
        xi = r_xi.first;

        // Check for convergence
        if (std::abs(r_mu.second - (-gev_log_likelihood(data, mu, sigma, xi))) < tol) {
            break;
        }
    }

    Eigen::Vector3d result;
    result << mu, sigma, xi;
    return result;
}

// Parallelized function to apply gev_mle to each column of a matrix
// [[Rcpp::export]]
Eigen::MatrixXd gev_mle_matrix(const Eigen::MatrixXd& data) {
    int n_stations = data.cols();
    Eigen::MatrixXd results(n_stations, 3);

    #pragma omp parallel for
    for (int i = 0; i < n_stations; ++i) {
        Eigen::VectorXd station_data = data.col(i);
        Eigen::Vector3d station_results = gev_mle(station_data);
        
        #pragma omp critical
        {
            results.row(i) = station_results;
        }
    }

    return results;
}