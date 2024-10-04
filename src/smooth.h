#ifndef SMOOTH_H
#define SMOOTH_H

#include <RcppEigen.h>
#include <vector>

// Data Structures
struct LatentField {
    std::vector<Eigen::VectorXd> parameters;
    LatentField(int n_parameters, int n_locations);
};

struct Hyperparameters {
    Eigen::VectorXd tau;
    Hyperparameters(int n_parameters);
};

struct ModelData {
    Eigen::VectorXd eta_hat;
    Eigen::SparseMatrix<double> Q_eta_y;
    Eigen::SparseMatrix<double> L_eta_y;
    Eigen::VectorXd b_eta_y;
    Eigen::SparseMatrix<double> Q_ICAR;
    Eigen::SparseMatrix<double> L_ICAR;
    int n_parameters;
    int n_locations;
    ModelData(int n_params, int n_locs);
};

struct MCMCResult {
    std::vector<LatentField> field_samples;
    std::vector<Hyperparameters> hyper_samples;
    MCMCResult(int n_iterations, int n_parameters, int n_locations);
};

// Function declarations
double p_hyper_cond_obs(const ModelData& data, const LatentField& field, const Hyperparameters& hyper);
double p_obs_cond_latent(const LatentField& field, const ModelData& data);
double p_latent_cond_hyper(const LatentField& field, const Hyperparameters& hyper, const ModelData& data);
double p_hyper(const Hyperparameters& hyper);
double p_latent_cond_obs_hyper(const LatentField& field, const ModelData& data, const Hyperparameters& hyper);
void sample_latent_field(LatentField& field, const Hyperparameters& hyper, const ModelData& data);
bool metropolis_hastings_step(Hyperparameters& hyper, const LatentField& field, const ModelData& data);
MCMCResult mcmc_smooth_step(const ModelData& data, int n_iterations);
Eigen::SparseMatrix<double> scale_cholesky(const Eigen::SparseMatrix<double>& L_ICAR, const Hyperparameters& hyper);
double compute_quadratic_form(const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& L);
void initialize_parameters(LatentField& field, Hyperparameters& hyper, const ModelData& data);

#endif // SMOOTH_H
