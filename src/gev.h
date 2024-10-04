#ifndef GEV_H
#define GEV_H

#include <RcppEigen.h>
#include <nlopt.hpp>

namespace GEV {

struct MLEResult {
    Eigen::Vector3d estimates;
    Eigen::Matrix3d hessian;
};

// GEV log-likelihood function
double gev_log_likelihood(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// Full gradient of GEV log-likelihood function
Eigen::Vector3d gev_log_likelihood_gradient(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// Full Hessian of GEV log-likelihood function
Eigen::Matrix3d gev_hessian(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// NLopt objective function
double gev_neg_log_likelihood_nlopt(unsigned n, const double* x, double* grad, void* data);

// MLE function using NLopt optimization
MLEResult mle(const Eigen::VectorXd& data);

} // namespace GEV

#endif // GEV_H