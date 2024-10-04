#ifndef GEV_H
#define GEV_H

#include <RcppEigen.h>
#include <boost/math/tools/minima.hpp>

namespace GEV {

struct MLEResult {
    Eigen::Vector3d estimates;
    Eigen::Matrix3d hessian;
};

// GEV log-likelihood function
double gev_log_likelihood(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// Individual gradient components
double gev_gradient_mu(const Eigen::Vector3d& params, const Eigen::VectorXd& data);
double gev_gradient_sigma(const Eigen::Vector3d& params, const Eigen::VectorXd& data);
double gev_gradient_xi(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// Full gradient of GEV log-likelihood function
Eigen::Vector3d gev_log_likelihood_gradient(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// Full Hessian of GEV log-likelihood function
Eigen::Matrix3d gev_hessian(const Eigen::Vector3d& params, const Eigen::VectorXd& data);

// Wrapper function for Boost optimization
struct gev_log_likelihood_functor {
    const Eigen::VectorXd& data;
    
    gev_log_likelihood_functor(const Eigen::VectorXd& data);
    
    double operator()(const Eigen::Vector3d& params) const;
};

// MLE function using Boost optimization
MLEResult mle(const Eigen::VectorXd& data);

} // namespace GEV

#endif // GEV_H