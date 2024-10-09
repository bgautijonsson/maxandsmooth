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
    std::vector<Eigen::SparseMatrix<double>> Q_diags;
    std::vector<Eigen::SparseMatrix<double>> Q_offdiags;
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
    std::vector<bool> accepted_samples;
    MCMCResult(int n_iterations, int n_parameters, int n_locations);
};

// Function declarations
MCMCResult mcmc_smooth_step(const ModelData& data, int n_iterations, int burn_in, int thin, int print_every, double proposal_sd);

#endif // SMOOTH_H
