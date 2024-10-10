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

// GEV log-likelihood function
dual gev_loglik(const VectorXdual& params, const VectorXd& data) {
    dual psi = params(0);
    dual tau = params(1);
    dual phi = params(2);

    dual mu = exp(psi);
    dual sigma = exp(psi + tau);
    dual xi = pow(1 + exp(-phi), -1);


    int n = data.size();
    dual loglik = 0;

    for(int i = 0; i < n; ++i) {
        dual z = (data(i) - mu) / sigma;

        if (xi < 1e-6) {
            loglik -= log(sigma) + z + exp(-z);
        } else {
            dual t = 1 + xi * z;
            loglik -= log(sigma) + (1.0 + 1.0 / xi) * log(t) + pow(t, -1.0 / xi);
        }

        
        
        
    }

    // Regularization
    // loglik -= pow(psi, 2); 

    return loglik;
}

// Function to calculate gradient
Eigen::VectorXd gev_gradient(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    VectorXdual param_dual = params.cast<dual>();
    VectorXdual grad;

    auto f = [&data](const VectorXdual& p) { return gev_loglik(p, data); };
    grad = gradient(f, wrt(param_dual), at(param_dual));

    return grad.cast<double>();
}

// Function to calculate log-likelihood
double gev_loglik_value(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    VectorXdual param_dual = params.cast<dual>();
    return val(gev_loglik(param_dual, data));
}

// Objective function for NLopt
double objective(unsigned n, const double* x, double* grad, void* f_data) {
    const Eigen::VectorXd* pData = reinterpret_cast<const Eigen::VectorXd*>(f_data);
    const Eigen::VectorXd& dataVec = *pData;
    Eigen::Vector3d params;
    for (int i = 0; i < 3; ++i) {
        params(i) = x[i];
    }

    double loglik = gev_loglik_value(params, dataVec);

    if (grad) {
        Eigen::Vector3d gradient = -gev_gradient(params, dataVec);
        for (int i = 0; i < 3; ++i) {
            grad[i] = gradient(i);
        }
    }

    return -loglik;  // Negative because we're minimizing
}

// [[Rcpp::export]]
Rcpp::List gev_mle(const Eigen::VectorXd& data) {
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

    Eigen::Vector3d mle_params(x[0], x[1], x[2]);
    Eigen::Matrix3d precision = gev_hessian(mle_params, data);
    Eigen::Vector6d lower_triangle;
    lower_triangle << precision(0,0), precision(1,0), precision(1,1), precision(2,0), precision(2,1), precision(2,2);

    return Rcpp::List::create(
        Rcpp::Named("params") = mle_params,
        Rcpp::Named("precision") = lower_triangle
    );
}

// Function to calculate Hessian
Eigen::Matrix3d gev_hessian(const Eigen::Vector3d& params, const Eigen::VectorXd& data) {
    VectorXdual param_dual = params.cast<dual>();
    Matrix3dual hess;

    auto f = [&data](const VectorXdual& p) { return gev_loglik(p, data); };
    hess = hessian(f, wrt(param_dual), at(param_dual));

    return -hess.cast<double>();  // Return negative Hessian as precision matrix
}

// [[Rcpp::export]]
Rcpp::List gev_mle_multiple(Eigen::MatrixXd& data) {
    int n_locations = data.cols();
    Eigen::MatrixXd results(n_locations, 3);
    std::vector<Eigen::SparseMatrix<double>> precision_blocks(6);
    
    for (int i = 0; i < 6; ++i) {
        precision_blocks[i] = Eigen::SparseMatrix<double>(n_locations, n_locations);
    }

    #pragma omp parallel for
    for (int i = 0; i < n_locations; ++i) {
        Rcpp::List mle_result = gev_mle(data.col(i));
        results.row(i) = Rcpp::as<Eigen::Vector3d>(mle_result["params"]);
        Eigen::Vector6d lower_triangle = Rcpp::as<Eigen::Vector6d>(mle_result["precision"]);
        
        for (int j = 0; j < 6; ++j) {
            precision_blocks[j].insert(i, i) = lower_triangle(j);
        }
    }

    for (int i = 0; i < 6; ++i) {
        precision_blocks[i].makeCompressed();
    }

    return Rcpp::List::create(
        Rcpp::Named("params") = results,
        Rcpp::Named("Q_psi_psi") = precision_blocks[0],
        Rcpp::Named("Q_tau_psi") = precision_blocks[1],
        Rcpp::Named("Q_tau_tau") = precision_blocks[2],
        Rcpp::Named("Q_phi_psi") = precision_blocks[3],
        Rcpp::Named("Q_phi_tau") = precision_blocks[4],
        Rcpp::Named("Q_phi_phi") = precision_blocks[5]
    );
}