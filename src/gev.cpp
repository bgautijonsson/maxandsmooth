#include "gev.h"
#include <cmath>
#include <limits>
#include <nlopt.hpp>

namespace GEV {

// Helper functions for link and inverse link
double inv_logit(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double logit(double x) {
    return std::log(x / (1.0 - x));
}

double gev_log_likelihood(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double psi = params[0];
    double tau = params[1];
    double phi = params[2];

    double mu = std::exp(psi);
    double sigma = std::exp(tau + psi);
    double xi = inv_logit(phi);

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

Eigen::Vector3d gev_log_likelihood_gradient(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double psi = params[0];
    double tau = params[1];
    double phi = params[2];

    double mu = std::exp(psi);
    double sigma = std::exp(tau + psi);
    double xi = inv_logit(phi);

    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();

    for (int i = 0; i < data.size(); ++i) {
        double x = data(i);
        double e2 = std::exp(-phi);
        double e3 = 1 + e2;
        double e4 = std::exp(psi + tau);
        double e7 = (x - mu) / (e3 * e4) + 1;

        // Gradient for psi
        gradient(0) -= 1 + x * (1 / std::pow(e7, e3) - (2 + e2) / e3) / (e7 * e4);

        // Gradient for tau
        double e6 = x - mu;
        double e9 = e6 / (e3 * e4) + 1;
        gradient(1) -= (1 / std::pow(e9, e3) - (2 + e2) / e3) * e6 / (e9 * e4) + 1;

        // Gradient for phi
        double e8 = e6 / (e3 * e4);
        e9 = e8 + 1;
        double e10 = 2 + e2;
        gradient(2) -= ((e3 * (1 / std::pow(e9, e3) - 1) * std::log1p(e8) - e6 / (std::pow(e9, e10) * e4)) * e3 + e10 * e6 / (e9 * e4)) * e2 / std::pow(e3, 2);
    }

    return gradient;
}

// These calculations are based on the symbolic differentiation of the GEV log-likelihood
// with respect to the transformed parameters shown in the file symbolic_diff_link_function.R
Eigen::Matrix3d gev_hessian(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    double psi = params[0];
    double tau = params[1];
    double phi = params[2];

    double mu = std::exp(psi);
    double sigma = std::exp(tau + psi);
    double xi = inv_logit(phi);

    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();

    for (int i = 0; i < data.size(); ++i) {
        double x = data(i);
        double e2 = std::exp(-phi);
        double e3 = 1 + e2;
        double e4 = std::exp(psi + tau);
        double e5 = e3 * e4;
        double e6 = mu;
        double e7 = x - e6;
        double e9 = e7 / e5 + 1;
        double e10 = std::pow(e5, 2);
        double e11 = e6 / e5;

        // Hessian: psi, psi
        hessian(0, 0) += x * (((1 / e5 - e5 / e10) * e7 + 1 - e11) * 
            (1 / std::pow(e9, e3) - (2 + e2) / e3) * e4 / std::pow(e9 * e4, 2) - 
            (e5 * e7 / e10 + e11) * e3 / (std::pow(e9, 3 + e2) * e4));

        // Hessian: psi, tau
        hessian(0, 1) += x * (((1 / e5 - e5 / e10) * e7 + 1) * 
            (1 / std::pow(e9, e3) - (2 + e2) / e3) * e4 / std::pow(e9 * e4, 2) - 
            std::pow(e3, 2) * e7 / (e10 * std::pow(e9, 3 + e2)));

        // Hessian: psi, phi
        double e8 = e7 / e5;
        hessian(0, 2) += x * ((std::pow(e9, e2) * e3 * e4 * e7 / e10 - 
            std::pow(e9, e3) * std::log1p(e8)) / std::pow(e9, 2 * e3) - 
            (1 - (2 + e2) / e3) / e3) / (e9 * e4);
        hessian(0, 2) += x * (1 / std::pow(e9, e3) - (2 + e2) / e3) * 
            std::pow(e4, 2) * e7 / (std::pow(e9 * e4, 2) * e10) * e2;

        // Hessian: tau, tau
        hessian(1, 1) += (((1 / e5 - e5 / e10) * e7 + 1) * 
            (1 / std::pow(e9, e3) - (2 + e2) / e3) * e4 / std::pow(e9 * e4, 2) - 
            std::pow(e3, 2) * e7 / (e10 * std::pow(e9, 3 + e2))) * e7;

        // Hessian: tau, phi
        hessian(1, 2) += ((std::pow(e9, e2) * e3 * e4 * e7 / e10 - 
            std::pow(e9, e3) * std::log1p(e8)) / std::pow(e9, 2 * e3) - 
            (1 - (2 + e2) / e3) / e3) / (e9 * e4);
        hessian(1, 2) += (1 / std::pow(e9, e3) - (2 + e2) / e3) * 
            std::pow(e4, 2) * e7 / (std::pow(e9 * e4, 2) * e10) * e2 * e7;

        // Hessian: phi, phi
        double e13 = std::log1p(e8);
        double e14 = e9 * e4;
        double e15 = std::pow(e9, 2 + e2);
        double e17 = 1 / std::pow(e9, e3) - 1;
        double e18 = e15 * e4;
        double e19 = e3 * e17;
        double phi_phi_term1 = ((std::pow(e9, e3) * (2 + e2) * e4 * e7 / e10 - e15 * e13) / 
            std::pow(e18, 2) + e19 / (e10 * e9)) * e4 * e7;
        double phi_phi_term2 = ((std::pow(e9, e2) * e3 * e4 * e7 / e10 - std::pow(e9, e3) * e13) * 
            e3 / std::pow(e9, 2 * e3 - e3) + 2) / std::pow(e9, e3) - 2;
        double phi_phi_term3 = (e17 / e14 - (2 + e2) * std::pow(e4, 2) * e7 / 
            (std::pow(e14, 2) * e10)) * e7;
        double phi_phi_term4 = ((e19 * e13 - e7 / e18) * e3 + (2 + e2) * e7 / e14) * 
            (1 - 2 * (e2 / e3));
        hessian(2, 2) -= ((phi_phi_term1 - phi_phi_term2 * e13) * e3 + phi_phi_term3) * e2 - 
            phi_phi_term4 * e2 / std::pow(e3, 2);
    }

    // Fill the lower triangle of the Hessian
    hessian(1, 0) = hessian(0, 1);
    hessian(2, 0) = hessian(0, 2);
    hessian(2, 1) = hessian(1, 2);

    return -hessian;  // Return negative Hessian for Fisher Information
}

double gev_neg_log_likelihood_nlopt(unsigned n, const double* x, double* grad, void* data) {
    const Eigen::VectorXd* pData = reinterpret_cast<const Eigen::VectorXd*>(data);
    const Eigen::VectorXd& dataVec = *pData;

    Eigen::Vector3d params;
    for (unsigned i = 0; i < n; ++i) {
        params[i] = x[i];
    }

    double log_lik = gev_log_likelihood(params, dataVec);

    if (std::isinf(log_lik) || std::isnan(log_lik)) {
        return std::numeric_limits<double>::max();
    }

    double neg_log_lik = -log_lik;

    if (grad) {
        Eigen::Vector3d gradient = -gev_log_likelihood_gradient(params, dataVec);
        if (gradient.hasNaN() || !gradient.allFinite()) {
            for (unsigned i = 0; i < n; ++i) {
                grad[i] = 0;
            }
        } else {
            for (unsigned i = 0; i < n; ++i) {
                grad[i] = gradient[i];
            }
        }
    }

    return neg_log_lik;
}

MLEResult mle(const Eigen::VectorXd& data) {
    MLEResult result;

    // Initial guesses for transformed parameters
    Eigen::Vector3d init_params(std::log(5.0), std::log(1.0) - std::log(5.0), logit(0.1));
    std::vector<double> x(init_params.data(), init_params.data() + init_params.size());

    // No bounds needed for transformed parameters
    nlopt::opt opt(nlopt::LD_LBFGS, 3);

    opt.set_min_objective(gev_neg_log_likelihood_nlopt, (void*)&data);

    opt.set_ftol_rel(1e-6);
    opt.set_xtol_rel(1e-6);
    opt.set_maxeval(1000);

    double minf;

    try {
        nlopt::result nlopt_result = opt.optimize(x, minf);

        for (int i = 0; i < 3; ++i) {
            init_params[i] = x[i];
        }

        result.estimates = init_params;
        result.hessian = gev_hessian(init_params, data);

    } catch (std::exception& e) {
        std::cerr << "NLopt failed: " << e.what() << std::endl;
        result.estimates = init_params;
        result.hessian = Eigen::Matrix3d::Zero();
    }

    return result;
}

} // namespace GEV