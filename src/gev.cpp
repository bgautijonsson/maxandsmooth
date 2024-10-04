#include "gev.h"
#include <cmath>
#include <limits>
#include <boost/math/tools/minima.hpp>

namespace GEV {

double gev_log_likelihood(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double mu = params[0];
    double sigma = params[1];
    double xi = params[2];
    double log_lik = 0.0;
    
    for (int i = 0; i < data.size(); ++i) {
        double z = (data(i) - mu) / sigma;
        double t = 1 + xi * z;
        if (t <= 0) {
            return std::numeric_limits<double>::lowest();
        }
        log_lik -= std::log(sigma) + (1 + 1/xi) * std::log(t) + std::pow(t, -1/xi);
    }
    return log_lik;
}

double gev_gradient_mu(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double mu = params[0];
    double sigma = params[1];
    double xi = params[2];
    double grad_mu = 0.0;
    
    for (int i = 0; i < data.size(); ++i) {
        double e1 = 1 + xi * (data(i) - mu) / sigma;
        double e2 = 1 / xi;
        grad_mu += (1 / std::pow(e1, e2) - xi * (1 + e2)) / (sigma * e1);
    }
    return -grad_mu;
}

double gev_gradient_sigma(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double mu = params[0];
    double sigma = params[1];
    double xi = params[2];
    double grad_sigma = 0.0;
    
    for (int i = 0; i < data.size(); ++i) {
        double e1 = data(i) - mu;
        double e2 = 1 + xi * e1 / sigma;
        double e3 = 1 / xi;
        grad_sigma += ((1 / std::pow(e2, e3) - xi * (1 + e3)) * e1 / (sigma * e2) + 1) / sigma;
    }
    return -grad_sigma;
}

double gev_gradient_xi(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double mu = params[0];
    double sigma = params[1];
    double xi = params[2];
    double grad_xi = 0.0;
    
    for (int i = 0; i < data.size(); ++i) {
        double e1 = data(i) - mu;
        double e3 = xi * e1 / sigma;
        double e4 = 1 + e3;
        double e5 = 1 / xi;
        double e6 = 1 + e5;
        grad_xi += ((1 / std::pow(e4, e5) - 1) * std::log1p(e3) / xi - e1 / (sigma * std::pow(e4, e6))) / xi + e6 * e1 / (sigma * e4);
    }
    return -grad_xi;
}

Eigen::Vector3d gev_log_likelihood_gradient(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    Eigen::Vector3d gradient;
    gradient << gev_gradient_mu(params, data),
                gev_gradient_sigma(params, data),
                gev_gradient_xi(params, data);
    return gradient;
}

Eigen::Matrix3d gev_hessian(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double mu = params[0];
    double sigma = params[1];
    double xi = params[2];
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    
    for (int i = 0; i < data.size(); ++i) {
        double x = data(i);
        double e1 = x - mu;
        double e2 = xi * e1 / sigma;
        double e3 = 1 + e2;
        double e4 = 1 / xi;
        double e5 = 1 + e4;
        double e6 = std::pow(e3, e4);
        double e7 = sigma * e3;
        double e8 = std::pow(e3, e5);
        double e9 = std::log1p(e2);
        
        // Hessian: mu, mu
        hessian(0, 0) += xi * (1 / (sigma * xi * std::pow(e3, e4 + 2)) + sigma * (1 / e6 - xi * e5) / std::pow(sigma * e3, 2)) / sigma;
        
        // Hessian: mu, sigma
        hessian(0, 1) += (1 / e6 - xi * e5) / std::pow(sigma * e3, 2) - e1 / (std::pow(sigma, 3) * std::pow(e3, e4 + 2));
        
        // Hessian: mu, xi
        hessian(0, 2) += (std::pow(e3, e4 - 1) * e1 / sigma - e6 * e9 / xi) / (xi * std::pow(e3, 2 / xi)) / e7 + (1 / e6 - xi * e5) * e1 / std::pow(e7, 2);
        
        // Hessian: sigma, sigma
        hessian(1, 1) += ((1 / e6 - xi * e5) * e1 / e7 + 1) / sigma + ((1 / e6 - xi * e5) / std::pow(e7, 2) - e1 / (std::pow(sigma, 3) * std::pow(e3, e4 + 2))) * e1;
        
        // Hessian: sigma, xi
        hessian(1, 2) += ((std::pow(e3, e4 - 1) * e1 / sigma - e6 * e9 / xi) / (xi * std::pow(e3, 2 / xi)) + 1) / e7 + (1 / e6 - xi * e5) * e1 / std::pow(e7, 2) * e1 / sigma;
        
        // Hessian: xi, xi
        hessian(2, 2) += -((e1 / e7 - 2 * (e9 / xi)) * (1 / e6 - 1) + e1 / e8 - (std::pow(e3, e4 - 1) * e1 / sigma - e6 * e9 / xi) * e9 / (xi * std::pow(e3, 2 / xi))) / xi + sigma * (e5 * e6 * e1 / sigma - e8 * e9 / std::pow(xi, 2)) * e1 / std::pow(e8, 2) / xi - (e5 * e1 / std::pow(e7, 2) + 1 / (sigma * std::pow(xi, 2) * e3)) * e1;
    }
    
    // Fill the lower triangle of the Hessian
    hessian(1, 0) = hessian(0, 1);
    hessian(2, 0) = hessian(0, 2);
    hessian(2, 1) = hessian(1, 2);
    
    return -hessian;  // Return negative Hessian for Fisher Information
}

gev_log_likelihood_functor::gev_log_likelihood_functor(const Eigen::VectorXd& data) : data(data) {}

double gev_log_likelihood_functor::operator()(const Eigen::Vector3d& params) const {
    return -gev_log_likelihood(params, data);
}

MLEResult mle(const Eigen::VectorXd& data) {
    MLEResult result;
    
    // Initial guesses for parameters
    Eigen::Vector3d init_params(5, 1, 0.1);
    
    // Set up the optimization problem
    gev_log_likelihood_functor f(data);
    
    // Example lower bounds
    double lower_bounds[3] = {0, 0, 0};
    // Example upper bounds
    double upper_bounds[3] = {INFINITY, INFINITY, 0.5};
    
    for (int i = 0; i < 3; ++i) {
        auto optimize_param = [&](double x) {
            Eigen::Vector3d params = init_params;
            params[i] = x;
            return f(params);
        };
        
        std::pair<double, double> result = boost::math::tools::brent_find_minima(
            optimize_param, 
            lower_bounds[i], 
            upper_bounds[i], 
            20);
        
        init_params[i] = result.first;
    }
    
    // Extract results
    result.estimates = init_params;
    
    // Compute Hessian using the closed-form solution
    result.hessian = gev_hessian(init_params, data);
    
    return result;
}

} // namespace GEV