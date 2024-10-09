#include "smooth.h"
#include <RcppEigen.h>
#include <cmath>
#include <random>
#include <chrono>

// Implement member functions
LatentField::LatentField(int n_parameters, int n_locations) : parameters(n_parameters, Eigen::VectorXd(n_locations)) {
    // Rcpp::Rcout << "LatentField constructor called with n_parameters: " << n_parameters << ", n_locations: " << n_locations << std::endl;
}

Hyperparameters::Hyperparameters(int n_parameters) : tau(n_parameters) {
    // Rcpp::Rcout << "Hyperparameters constructor called with n_parameters: " << n_parameters << std::endl;
}

ModelData::ModelData(int n_params, int n_locs) : 
    eta_hat(n_params * n_locs),
    Q_eta_y(n_params * n_locs, n_params * n_locs),
    L_eta_y(n_params * n_locs, n_params * n_locs),
    Q_diags(n_params),
    b_eta_y(n_params * n_locs),
    Q_ICAR(n_locs, n_locs),
    L_ICAR(n_locs, n_locs),
    n_parameters(n_params),
    n_locations(n_locs) {
    // Rcpp::Rcout << "ModelData constructor called with n_params: " << n_params << ", n_locs: " << n_locs << std::endl;
}

struct BlockCholeskyFactors {
    // Cholesky factors for the diagonal blocks
    Eigen::SparseMatrix<double> L11;
    Eigen::SparseMatrix<double> L22;
    Eigen::SparseMatrix<double> L33;

    // Off-diagonal blocks
    Eigen::SparseMatrix<double> L21;
    Eigen::SparseMatrix<double> L31;
    Eigen::SparseMatrix<double> L32;

    // Number of locations (size of each block)
    int n_locations;
};



MCMCResult::MCMCResult(int n_iterations, int n_parameters, int n_locations)
    : field_samples(n_iterations, LatentField(n_parameters, n_locations)),
      hyper_samples(n_iterations, Hyperparameters(n_parameters)),
      accepted_samples(n_iterations, false) {
    // Rcpp::Rcout << "MCMCResult constructor called with n_iterations: " << n_iterations 
    //             << ", n_parameters: " << n_parameters << ", n_locations: " << n_locations << std::endl;
}

double compute_quadratic_form(const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& L) {
    // Rcpp::Rcout << "Entering compute_quadratic_form function" << std::endl;
    Eigen::VectorXd y = L.triangularView<Eigen::Lower>().transpose() * x;
    double result = y.squaredNorm();
    // Rcpp::Rcout << "Exiting compute_quadratic_form function with result: " << result << std::endl;
    return result;
}

double p_obs_cond_latent(const ModelData& data) {
    
    double quadratic = compute_quadratic_form(data.eta_hat, data.L_eta_y);
    double log_det = 2 * data.L_eta_y.diagonal().array().log().sum();
    
    double result = -0.5 * (quadratic + log_det + data.n_parameters * data.n_locations * std::log(2 * M_PI));
    // Rcpp::Rcout << "Exiting p_obs_cond_latent function with result: " << result << std::endl;
    return result;
}

double p_latent_cond_hyper(const Hyperparameters& hyper, const ModelData& data) {
    // Rcpp::Rcout << "Entering p_latent_cond_hyper function" << std::endl;
    double log_prior = 0.0;
    
    for (int p = 0; p < data.n_parameters; ++p) {
        log_prior += (data.n_locations - 1) * std::log(hyper.tau[p]) / 2;
    }
    
    // Rcpp::Rcout << "Exiting p_latent_cond_hyper function with result: " << log_prior << std::endl;
    return log_prior;
}

double p_hyper(const Hyperparameters& hyper) {
    // Rcpp::Rcout << "Entering p_hyper function" << std::endl;
    double log_prior = 0.0;
    
    for (int p = 0; p < hyper.tau.size(); ++p) {
        double log_prec = std::log(hyper.tau[p]);
        double log_sd = - 0.5 * log_prec;
        double sd = std::exp(log_sd);
        log_prior += log_sd - sd;
    }
    
    // Rcpp::Rcout << "Exiting p_hyper function with result: " << log_prior << std::endl;
    return log_prior;
}

double p_latent_cond_obs_hyper(const LatentField& field, const ModelData& data, const Hyperparameters& hyper) {
    // Rcpp::Rcout << "Entering p_latent_cond_obs_hyper function" << std::endl;
    Eigen::SparseMatrix<double> tau_diag(data.n_parameters, data.n_parameters);
    tau_diag.setIdentity();
    for (int i = 0; i < data.n_parameters; ++i) {
        tau_diag.coeffRef(i, i) = hyper.tau(i);
    }
    Eigen::SparseMatrix<double> Q_latent = Eigen::kroneckerProduct(tau_diag, data.Q_ICAR);

    Eigen::SparseMatrix<double> Q_combined = Q_latent + data.Q_eta_y;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> cholsolver(Q_combined);
    Eigen::SparseMatrix<double> L_combined = cholsolver.matrixL();
    Eigen::VectorXd u = cholsolver.matrixL().solve(data.b_eta_y);
    double quadratic = u.squaredNorm();
    double log_det = 2 * L_combined.diagonal().array().log().sum();
    double log_prob = -0.5 * (quadratic + log_det + data.n_parameters * data.n_locations * std::log(2 * M_PI));
    
    // Rcpp::Rcout << "Exiting p_latent_cond_obs_hyper function with result: " << log_prob << std::endl;
    return log_prob;
}



// Core Functions
double p_hyper_cond_obs(const ModelData& data, const LatentField& field, const Hyperparameters& hyper) {
    // Rcpp::Rcout << "Entering p_hyper_cond_obs function" << std::endl;
    double marginal_log_dens = p_hyper(hyper) + 
        p_obs_cond_latent(data) + 
        p_latent_cond_hyper(hyper, data) -
        p_latent_cond_obs_hyper(field, data, hyper);
    // Rcpp::Rcout << "Exiting p_hyper_cond_obs function with result: " << marginal_log_dens << std::endl;
    return marginal_log_dens;
}

void initialize_parameters(LatentField& field, Hyperparameters& hyper, const ModelData& data) {
    // Rcpp::Rcout << "Entering initialize_parameters function" << std::endl;
    for (int p = 0; p < data.n_parameters; ++p) {
        field.parameters[p] = data.eta_hat.segment(p * data.n_locations, data.n_locations);
    }
    
    for (int p = 0; p < data.n_parameters; ++p) {
        Eigen::VectorXd param = field.parameters[p];
        hyper.tau(p) = 1.0;
    }
    // Rcpp::Rcout << "Exiting initialize_parameters function" << std::endl;
}

#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

BlockCholeskyFactors block_cholesky_decomposition(
    const ModelData& data,
    const Hyperparameters& hyper) {

    int n_locations = data.n_locations;

    // Diagonal blocks
    Eigen::SparseMatrix<double> D_mu = data.Q_diags[0];     // D_mu
    Eigen::SparseMatrix<double> D_sigma = data.Q_diags[1];  // D_sigma
    Eigen::SparseMatrix<double> D_xi = data.Q_diags[2];     // D_xi

    // Off-diagonal blocks
    Eigen::SparseMatrix<double> Q_mu_sigma = data.Q_offdiags[0];  // Q_{mu_sigma}
    Eigen::SparseMatrix<double> Q_mu_xi = data.Q_offdiags[1];     // Q_{mu_xi}
    Eigen::SparseMatrix<double> Q_sigma_xi = data.Q_offdiags[2];  // Q_{sigma_xi}

    // Compute Q11 and its Cholesky factor L11
    Eigen::SparseMatrix<double> Q11 = hyper.tau(0) * data.Q_ICAR + D_mu;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt11(Q11);
    Eigen::SparseMatrix<double> L11 = llt11.matrixL();

    // Compute L21 = L11^{-T} * Q_{mu_sigma}^T
    Eigen::SparseMatrix<double> Q21_T = Q_mu_sigma.transpose();
    Eigen::SparseMatrix<double> L11_T = L11.transpose();
    Eigen::SparseMatrix<double> L21_T = L11_T.triangularView<Eigen::Upper>().solve(Q21_T);
    Eigen::SparseMatrix<double> L21 = L21_T.transpose();

    // Compute L31 = L11^{-T} * Q_{mu_xi}^T
    Eigen::SparseMatrix<double> Q31_T = Q_mu_xi.transpose();
    Eigen::SparseMatrix<double> L31_T = L11_T.triangularView<Eigen::Upper>().solve(Q31_T);
    Eigen::SparseMatrix<double> L31 = L31_T.transpose();

    // Compute S22 = Q22 - L21 * L21^T
    Eigen::SparseMatrix<double> Q22 = hyper.tau(1) * data.Q_ICAR + D_sigma;
    Eigen::SparseMatrix<double> S22 = Q22 - (L21 * L21.transpose());

    // Compute L22
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt22(S22);
    Eigen::SparseMatrix<double> L22 = llt22.matrixL();

    // Compute L32 = L22^{-T} * (Q_{sigma_xi}^T - L31 * L21^T)
    Eigen::SparseMatrix<double> temp = Q_sigma_xi.transpose() - L31 * L21.transpose();
    Eigen::SparseMatrix<double> L22_T = L22.transpose();
    Eigen::SparseMatrix<double> L32_T = L22_T.triangularView<Eigen::Upper>().solve(temp);
    Eigen::SparseMatrix<double> L32 = L32_T.transpose();

    // Compute S33 = Q33 - L31 * L31^T - L32 * L32^T
    Eigen::SparseMatrix<double> Q33 = hyper.tau(2) * data.Q_ICAR + D_xi;
    Eigen::SparseMatrix<double> S33 = Q33 - (L31 * L31.transpose()) - (L32 * L32.transpose());

    // Compute L33
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt33(S33);
    Eigen::SparseMatrix<double> L33 = llt33.matrixL();

    // Since we need L for sampling, we can store L11, L21, L31, L22, L32, L33 separately
    // and use them in block forward and backward substitution when needed.

    // Return the components of L as a struct or class (or assemble if necessary)
    // For this example, we'll assume we return a custom structure

    BlockCholeskyFactors L_factors;
    L_factors.L11 = L11;
    L_factors.L21 = L21;
    L_factors.L31 = L31;
    L_factors.L22 = L22;
    L_factors.L32 = L32;
    L_factors.L33 = L33;

    return L_factors;
}

Eigen::VectorXd block_forward_substitution(
    const BlockCholeskyFactors& L_factors,
    const Eigen::VectorXd& b) {

    int n = L_factors.n_locations;

    // Split b into its corresponding blocks
    Eigen::VectorXd b1 = b.segment(0, n);
    Eigen::VectorXd b2 = b.segment(n, n);
    Eigen::VectorXd b3 = b.segment(2 * n, n);

    // Initialize y vectors
    Eigen::VectorXd y1(n), y2(n), y3(n);

    // Solve L11 * y1 = b1
    y1 = L_factors.L11.triangularView<Eigen::Lower>().solve(b1);

    // Compute right-hand side for y2
    Eigen::VectorXd rhs2 = b2 - L_factors.L21 * y1;

    // Solve L22 * y2 = rhs2
    y2 = L_factors.L22.triangularView<Eigen::Lower>().solve(rhs2);

    // Compute right-hand side for y3
    Eigen::VectorXd rhs3 = b3 - L_factors.L31 * y1 - L_factors.L32 * y2;

    // Solve L33 * y3 = rhs3
    y3 = L_factors.L33.triangularView<Eigen::Lower>().solve(rhs3);

    // Concatenate y1, y2, y3 into a single vector y
    Eigen::VectorXd y(3 * n);
    y << y1, y2, y3;

    return y;
}


Eigen::VectorXd block_backward_substitution(
    const BlockCholeskyFactors& L_factors,
    const Eigen::VectorXd& y) {

    int n = L_factors.n_locations;

    // Split y into its corresponding blocks
    Eigen::VectorXd y1 = y.segment(0, n);
    Eigen::VectorXd y2 = y.segment(n, n);
    Eigen::VectorXd y3 = y.segment(2 * n, n);

    // Initialize x vectors
    Eigen::VectorXd x1(n), x2(n), x3(n);

    // Solve L33^T * x3 = y3
    x3 = L_factors.L33.transpose().triangularView<Eigen::Upper>().solve(y3);

    // Compute right-hand side for x2
    Eigen::VectorXd rhs2 = y2 - L_factors.L32.transpose() * x3;

    // Solve L22^T * x2 = rhs2
    x2 = L_factors.L22.transpose().triangularView<Eigen::Upper>().solve(rhs2);

    // Compute right-hand side for x1
    Eigen::VectorXd rhs1 = y1 - L_factors.L21.transpose() * x2 - L_factors.L31.transpose() * x3;

    // Solve L11^T * x1 = rhs1
    x1 = L_factors.L11.transpose().triangularView<Eigen::Upper>().solve(rhs1);

    // Concatenate x1, x2, x3 into a single vector x
    Eigen::VectorXd x(3 * n);
    x << x1, x2, x3;

    return x;
}



LatentField sample_latent_field(const Hyperparameters& hyper, const ModelData& data) {
    // Rcpp::Rcout << "Entering sample_latent_field function" << std::endl;
    LatentField proposed_field(data.n_parameters, data.n_locations);
    
    // Compute block Cholesky decomposition
    BlockCholeskyFactors L_factors = block_cholesky_decomposition(data, hyper);
    
    // Solve L u = b (forward substitution)
    Eigen::VectorXd u = block_forward_substitution(L_factors, data.b_eta_y);
    // Solve L^T mu = u (backward substitution)
    Eigen::VectorXd mu_post = block_backward_substitution(L_factors, u);
    
    // Sample standard normal random vector
    Eigen::VectorXd z = Eigen::VectorXd::NullaryExpr(data.b_eta_y.size(), [&]() { return R::rnorm(0,1); });
    
    // Solve L delta = z (forward substitution)
    Eigen::VectorXd delta = block_forward_substitution(L_factors, z);

    // Compute sample
    Eigen::VectorXd eta_sample = mu_post + delta;
    
    // Assign sampled values to proposed field
    for (int p = 0; p < data.n_parameters; ++p) {
        proposed_field.parameters[p] = eta_sample.segment(p * data.n_locations, data.n_locations);
    }
    
    // Rcpp::Rcout << "Exiting sample_latent_field function" << std::endl;
    return proposed_field;
}

bool metropolis_hastings_step(Hyperparameters& hyper, LatentField& field, const ModelData& data, double proposal_sd) {
    // Rcpp::Rcout << "Entering metropolis_hastings_step function" << std::endl;
    Hyperparameters proposed_hyper = hyper;
    
    std::normal_distribution<> normal(0, proposal_sd);
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    for (int p = 0; p < data.n_parameters; ++p) {
        proposed_hyper.tau(p) *= std::exp(normal(generator));
    }
    
    LatentField proposed_field = sample_latent_field(proposed_hyper, data);
    
    double proposed_marginal_log_dens = p_hyper_cond_obs(data, proposed_field, proposed_hyper);
    double current_marginal_log_dens = p_hyper_cond_obs(data, field, hyper);
    double log_ratio = proposed_marginal_log_dens - current_marginal_log_dens;
    std::uniform_real_distribution<> uniform(0, 1);
    double log_uniform = std::log(uniform(generator));
    bool accepted = log_uniform < log_ratio;
    // Rcpp::Rcout << "log_ratio: " << log_ratio << " uniform: " << log_uniform << "accepted: " << accepted << std::endl;
    if (accepted) {
        hyper = proposed_hyper;
        field = proposed_field;
    }
    // Rcpp::Rcout << "Exiting metropolis_hastings_step function. Accepted: " << (accepted ? "true" : "false") << std::endl;
    return accepted;
}

MCMCResult mcmc_smooth_step(const ModelData& data, int n_iterations, int burn_in, int thin, int print_every, double proposal_sd) {
    // Rcpp::Rcout << "Entering mcmc_smooth_step function" << std::endl;
    int total_iterations = n_iterations * thin + burn_in;
    MCMCResult result(n_iterations, data.n_parameters, data.n_locations);
    LatentField field(data.n_parameters, data.n_locations);
    Hyperparameters hyper(data.n_parameters);
    
    initialize_parameters(field, hyper, data);

    for (int iter = 0; iter < total_iterations; ++iter) {
        // Rcpp::Rcout << "MCMC iteration: " << iter << std::endl;
        bool accepted = metropolis_hastings_step(hyper, field, data, proposal_sd);
        
        if (iter >= burn_in && (iter - burn_in) % thin == 0) {
            int sample_index = (iter - burn_in) / thin;
            result.field_samples[sample_index] = field;
            result.hyper_samples[sample_index] = hyper;
            result.accepted_samples[sample_index] = accepted;
        }

        if (iter % print_every == 0) {
            Rcpp::Rcout << "Iteration: " << iter << std::endl;
        }
    }
    
    // Rcpp::Rcout << "Exiting mcmc_smooth_step function" << std::endl;
    return result;
}