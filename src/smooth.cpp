#include "smooth.h"
#include <RcppEigen.h>
#include <cmath>
#include <random>
#include <chrono>

// Implement member functions
LatentField::LatentField(int n_parameters, int n_locations) : parameters(n_parameters, Eigen::VectorXd(n_locations)) {}

Hyperparameters::Hyperparameters(int n_parameters) : tau(n_parameters) {}

ModelData::ModelData(int n_params, int n_locs) : 
    eta_hat(n_params * n_locs),
    Q_eta_y(n_params * n_locs, n_params * n_locs),
    L_eta_y(n_params * n_locs, n_params * n_locs),
    b_eta_y(n_params * n_locs),
    Q_ICAR(n_locs, n_locs),
    L_ICAR(n_locs, n_locs),
    n_parameters(n_params),
    n_locations(n_locs) {}

MCMCResult::MCMCResult(int n_iterations, int n_parameters, int n_locations)
    : field_samples(n_iterations, LatentField(n_parameters, n_locations)),
      hyper_samples(n_iterations, Hyperparameters(n_parameters)) {}

// Core Functions
double p_hyper_cond_obs(const ModelData& data, const LatentField& field, const Hyperparameters& hyper) {
  double marginal_log_dens = p_hyper(hyper) + 
    p_obs_cond_latent(field, data) + 
    p_latent_cond_hyper(field, hyper, data) -
    p_latent_cond_obs_hyper(field, data, hyper);

  return marginal_log_dens;
}

// Log density of the data given the latent field, i.e. the likelihood
double p_obs_cond_latent(const LatentField& field, const ModelData& data) {
  Eigen::VectorXd diff(data.n_parameters * data.n_locations);
  for (int p = 0; p < data.n_parameters; ++p) {
    diff.segment(p * data.n_locations, data.n_locations) = field.parameters[p] - data.eta_hat.segment(p * data.n_locations, data.n_locations);
  }
  
  double quadratic = compute_quadratic_form(diff, data.L_eta_y);
  double log_det = 2 * data.L_eta_y.diagonal().array().log().sum();
  
  return -0.5 * (quadratic + log_det + data.n_parameters * data.n_locations * std::log(2 * M_PI));
}

// Log density of the latent field given the hyperparameters
// We can calculate this assuming any value for the field, i.e. x = 0
// Thus the density simply becomes tau^((n-1)/2) * 1 for all parameters
double p_latent_cond_hyper(const LatentField& field, const Hyperparameters& hyper, const ModelData& data) {
  double log_prior = 0.0;
  
  for (int p = 0; p < data.n_parameters; ++p) {
    log_prior += (data.n_locations - 1) * std::log(hyper.tau[p]) / 2;
  }
  
  return log_prior;
}

// Log density of hyperparameters
double p_hyper(const Hyperparameters& hyper) {
  double log_prior = 0.0;
  
  // Assuming a Gamma(a, b) prior for each tau
  double a = 1.0; // shape parameter
  double b = 0.00005; // rate parameter
  
  for (int p = 0; p < hyper.tau.size(); ++p) {
    log_prior += (a - 1) * std::log(hyper.tau[p]) - b * hyper.tau[p] - a * std::log(b) - std::lgamma(a);
  }
  
  return log_prior;
}

// Log density of the latent field given the observations and hyperparameters
double p_latent_cond_obs_hyper(const LatentField& field, const ModelData& data, const Hyperparameters& hyper) {
  Eigen::SparseMatrix<double> tau_diag(data.n_parameters, data.n_parameters);
  tau_diag.setIdentity();
  for (int i = 0; i < data.n_parameters; ++i) {
      tau_diag.coeffRef(i, i) = hyper.tau(i);
  }
  Eigen::SparseMatrix<double> Q_latent = Eigen::kroneckerProduct(tau_diag, data.Q_ICAR);

  // Rest of the function remains the same
  Eigen::SparseMatrix<double> Q_combined = Q_latent + data.Q_eta_y;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> cholsolver(Q_combined);
  Eigen::SparseMatrix<double> L_combined = cholsolver.matrixL();
  Eigen::VectorXd mu = cholsolver.matrixL().solve(cholsolver.matrixL().transpose().solve(data.b_eta_y));
  double quadratic = compute_quadratic_form(data.eta_hat - mu, L_combined);
  double log_det = 2 * L_combined.diagonal().array().log().sum();
  double log_prob = -0.5 * (quadratic + log_det + data.n_parameters * data.n_locations * std::log(2 * M_PI));
  
  return log_prob;
}

void sample_latent_field(LatentField& field, const Hyperparameters& hyper, const ModelData& data) {
  Eigen::SparseMatrix<double> tau_diag(data.n_parameters, data.n_parameters);
  tau_diag.setIdentity();
  for (int i = 0; i < data.n_parameters; ++i) {
      tau_diag.coeffRef(i, i) = hyper.tau(i);
  }
  Eigen::SparseMatrix<double> Q_latent = Eigen::kroneckerProduct(tau_diag, data.Q_ICAR);

  Eigen::SparseMatrix<double> Q_combined = Q_latent + data.Q_eta_y;
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> chol(Q_combined);
  Eigen::VectorXd mu = data.Q_eta_y * data.eta_hat;
  
  // Generate standard normal samples
  Eigen::VectorXd z(data.n_parameters * data.n_locations);
  std::normal_distribution<> normal(0, 1);
  std::default_random_engine generator;
  for (int i = 0; i < z.size(); ++i) {
    z(i) = normal(generator);
  }
  
  // Samples from the system
  Eigen::VectorXd sampled_field = mu + chol.solve(z);
  
  // Update the field parameters
  for (int p = 0; p < data.n_parameters; ++p) {
    field.parameters[p] = sampled_field.segment(p * data.n_locations, data.n_locations);
  }
}

bool metropolis_hastings_step(Hyperparameters& hyper, const LatentField& field, const ModelData& data) {
  Hyperparameters proposed_hyper = hyper;
  
  // Propose new hyperparameters using a random walk on log scale
  std::normal_distribution<> normal(0, 0.1);
  std::default_random_engine generator;
  for (int p = 0; p < data.n_parameters; ++p) {
    proposed_hyper.tau(p) = std::exp(std::log(hyper.tau(p)) + normal(generator));
  }
  
  // Compute acceptance ratio as
  // p(hyper) + p(data | latent, hyper) + p(latent | hyper) - p(latent | data, hyper)
  double proposed_marginal_log_dens = p_hyper_cond_obs(data, field, proposed_hyper);
  double current_marginal_log_dens = p_hyper_cond_obs(data, field, hyper);
  double log_ratio = proposed_marginal_log_dens - current_marginal_log_dens;
  // Accept or reject
  std::uniform_real_distribution<> uniform(0, 1);
  if (std::log(uniform(generator)) < log_ratio) {
    hyper = proposed_hyper;
    return true;
  }
  return false;
}


MCMCResult mcmc_smooth_step(const ModelData& data, int n_iterations) {
  MCMCResult result(n_iterations, data.n_parameters, data.n_locations);
  LatentField field(data.n_parameters, data.n_locations);
  Hyperparameters hyper(data.n_parameters);
  
  // Initialize field and hyperparameters
  initialize_parameters(field, hyper, data);
  
  // MCMC loop
  for (int iter = 0; iter < n_iterations; ++iter) {
    sample_latent_field(field, hyper, data);
    bool accepted = metropolis_hastings_step(hyper, field, data);
    
    // Store samples
    result.field_samples[iter] = field;
    result.hyper_samples[iter] = hyper;

    if (iter % 100 == 0) {
      Rcpp::Rcout << "Iteration: " << iter << std::endl;
    }
  }
  
  return result;
}

// Utility Functions
// Scale the Cholesky decomposition of the ICAR precision matrix by the relevant hyperparameters
Eigen::SparseMatrix<double> scale_cholesky(const Eigen::SparseMatrix<double>& L_ICAR, const Hyperparameters& hyper) {
  Eigen::SparseMatrix<double> L_scaled = L_ICAR;
  for (int k = 0; k < L_ICAR.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(L_ICAR, k); it; ++it) {
      L_scaled.coeffRef(it.row(), it.col()) *= std::sqrt(hyper.tau(k));
    }
  }
  return L_scaled;
}

// Compute quadratic forms efficiently using the Cholesky decomposition
double compute_quadratic_form(const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& L) {
  Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(x);
  return y.squaredNorm();
}

void initialize_parameters(LatentField& field, Hyperparameters& hyper, const ModelData& data) {
  // Initialize field
  for (int p = 0; p < data.n_parameters; ++p) {
    field.parameters[p] = data.eta_hat.segment(p * data.n_locations, data.n_locations);
  }
  
  // Initialize hyperparameters
  for (int p = 0; p < data.n_parameters; ++p) {
    Eigen::VectorXd param = field.parameters[p];
    double mean = param.mean();
    double variance = (param.array() - mean).square().sum() / (param.size() - 1);
    hyper.tau(p) = 1.0 / variance;
  }
}