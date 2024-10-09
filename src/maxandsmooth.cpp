#include <RcppEigen.h>
#include <vector>
#include "max.h"
#include "smooth.h"
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List prepare_mle_data(const Eigen::MatrixXd& estimates, const Rcpp::List& hessians) {
    // Rcpp::Rcout << "Entering prepare_mle_data function" << std::endl;
    int n_stations = estimates.rows();
    int n_params = estimates.cols();
    int total_params = n_stations * n_params;

    // Rcpp::Rcout << "Creating all_params vector" << std::endl;
    // Create the vector of all parameters, reordered
    Eigen::VectorXd all_params(total_params);
    for (int p = 0; p < n_params; ++p) {
        for (int i = 0; i < n_stations; ++i) {
            all_params[p * n_stations + i] = estimates(i, p);
        }
    }

    // Rcpp::Rcout << "Creating full_hessian matrix" << std::endl;
    // Create the reordered sparse matrix of negative Hessians
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(total_params * n_params);  // Approximate size

    for (int p1 = 0; p1 < n_params; ++p1) {
        for (int p2 = 0; p2 < n_params; ++p2) {
            for (int i = 0; i < n_stations; ++i) {
                Eigen::MatrixXd station_hessian = Rcpp::as<Eigen::MatrixXd>(hessians[i]);
                double value = station_hessian(p1, p2);
                if (value != 0) {
                    tripletList.push_back(T(p1 * n_stations + i, p2 * n_stations + i, value));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> full_hessian(total_params, total_params);
    full_hessian.setFromTriplets(tripletList.begin(), tripletList.end());

    // Create diagonal matrices Q11, Q12, Q13, Q22, Q23, Q33
    std::vector<Eigen::SparseMatrix<double>> Q_diags(3);
    std::vector<Eigen::SparseMatrix<double>> Q_offdiags(3);
    for (int i = 0; i < 3; ++i) {
        Q_diags[i] = Eigen::SparseMatrix<double>(n_stations, n_stations);
        Q_diags[i].reserve(Eigen::VectorXi::Constant(n_stations, 1));
    }

    for (int i = 0; i < n_stations; ++i) {
        Eigen::MatrixXd station_hessian = Rcpp::as<Eigen::MatrixXd>(hessians[i]);
        Q_diags[0].insert(i, i) = station_hessian(0, 0);  // Q11: mu-mu
        Q_offdiags[0].insert(i, i) = station_hessian(0, 1);  // Q12: mu-sigma
        Q_offdiags[0].insert(i, i) = station_hessian(0, 2);  // Q13: mu-xi
        Q_diags[1].insert(i, i) = station_hessian(1, 1);  // Q22: sigma-sigma
        Q_offdiags[1].insert(i, i) = station_hessian(1, 2);  // Q23: sigma-xi
        Q_diags[2].insert(i, i) = station_hessian(2, 2);  // Q33: xi-xi
    }

    // Rcpp::Rcout << "Exiting prepare_mle_data function" << std::endl;
    // Return the results as a list
    Rcpp::List result;
    result["all_params"] = all_params;
    result["full_hessian"] = full_hessian;
    result["Q_diags"] = Rcpp::wrap(Q_diags);
    result["Q_offdiags"] = Rcpp::wrap(Q_offdiags);

    return result;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> create_q_icar(int x_dim, int y_dim) {
    // Rcpp::Rcout << "Entering create_q_icar function" << std::endl;
    // Create the 1D random walk matrix
    Eigen::SparseMatrix<double> Q1(x_dim, x_dim);
    Eigen::SparseMatrix<double> Q2(y_dim, y_dim);
    Q1.reserve(Eigen::VectorXi::Constant(x_dim, 3));
    Q2.reserve(Eigen::VectorXi::Constant(y_dim, 3));

    for (int i = 0; i < x_dim; ++i) {
        if (i > 0) Q1.insert(i, i-1) = -1;
        Q1.insert(i, i) = 2;
        if (i < x_dim - 1) Q1.insert(i, i+1) = -1;
    }

    for (int i = 0; i < y_dim; ++i) {
        if (i > 0) Q2.insert(i, i-1) = -1;
        Q2.insert(i, i) = 2;
        if (i < y_dim - 1) Q2.insert(i, i+1) = -1;
    }

    // Define identity matrix I
    Eigen::SparseMatrix<double> I1(x_dim, x_dim);
    I1.setIdentity();
    Eigen::SparseMatrix<double> I2(y_dim, y_dim);
    I2.setIdentity();

    // Calculate sum of kronecker products
    Eigen::SparseMatrix<double> Q_ICAR = Eigen::kroneckerProduct(Q1, I2) + Eigen::kroneckerProduct(I1, Q2);

    // Rcpp::Rcout << "Exiting create_q_icar function" << std::endl;
    return Q_ICAR;
}

// [[Rcpp::export]]
Rcpp::List maxandsmooth_cpp(const Eigen::MatrixXd& data, int x_dim, int y_dim, const std::string& family, int n_iterations = 1000, int burn_in = 1000, int thin = 10, int print_every = 100, double proposal_sd = 0.1) {
    // Rcpp::Rcout << "Entering maxandsmooth_cpp function" << std::endl;

    // Rcpp::Rcout << "Step 1: Performing MLE (Max step)" << std::endl;
    Rcpp::List mle_result = max_cpp(data, family);
    // Rcpp::Rcout << "MLE step completed" << std::endl;
    Eigen::MatrixXd estimates = mle_result["estimates"];
    Rcpp::List hessians = mle_result["hessians"];

    // Rcpp::Rcout << "Step 2: Preparing data for smoothing" << std::endl;
    Rcpp::List prepared_data = prepare_mle_data(estimates, hessians);
    // Rcpp::Rcout << "Data preparation completed" << std::endl;
    Eigen::VectorXd all_params = prepared_data["all_params"];
    Eigen::SparseMatrix<double> full_hessian = prepared_data["full_hessian"];

    // Rcpp::Rcout << "Step 3: Setting up ModelData for smoothing" << std::endl;
    int n_stations = data.cols();
    int n_params = estimates.cols();
    ModelData model_data(n_params, n_stations);
    model_data.eta_hat = all_params;
    model_data.Q_eta_y = full_hessian;

    // Convert Rcpp::List to std::vector<Eigen::SparseMatrix<double>>
    Rcpp::List Q_diags_list = prepared_data["Q_diags"];
    model_data.Q_diags.resize(Q_diags_list.size());
    for (int i = 0; i < Q_diags_list.size(); ++i) {
        model_data.Q_diags[i] = Rcpp::as<Eigen::SparseMatrix<double>>(Q_diags_list[i]);
    }

    Rcpp::List Q_offdiags_list = prepared_data["Q_offdiags"];
    model_data.Q_offdiags.resize(Q_offdiags_list.size());
    for (int i = 0; i < Q_offdiags_list.size(); ++i) {
        model_data.Q_offdiags[i] = Rcpp::as<Eigen::SparseMatrix<double>>(Q_offdiags_list[i]);
    }

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> full_chol(full_hessian);
    model_data.L_eta_y = full_chol.matrixL();
    model_data.b_eta_y = model_data.Q_eta_y * model_data.eta_hat;

    // Rcpp::Rcout << "Creating Q_ICAR matrix" << std::endl;
    model_data.Q_ICAR = create_q_icar(x_dim, y_dim);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> icar_chol(model_data.Q_ICAR);
    model_data.L_ICAR = icar_chol.matrixL();

    // Rcpp::Rcout << "Step 4: Performing MCMC smoothing" << std::endl;
    MCMCResult mcmc_result = mcmc_smooth_step(model_data, n_iterations, burn_in, thin, print_every, proposal_sd);

    // Rcpp::Rcout << "Step 5: Preparing output" << std::endl;
    int total_params = n_stations * n_params;
    Eigen::MatrixXd smoothed_samples(n_iterations, total_params);
    Eigen::MatrixXd hyper_samples(n_iterations, n_params);

    for (int i = 0; i < n_iterations; ++i) {
        Eigen::VectorXd sample(total_params);
        for (int p = 0; p < n_params; ++p) {
            sample.segment(p * n_stations, n_stations) = mcmc_result.field_samples[i].parameters[p];
            hyper_samples(i, p) = mcmc_result.hyper_samples[i].tau(p);
        }
        smoothed_samples.row(i) = sample;
    }

    // Rcpp::Rcout << "Exiting maxandsmooth_cpp function" << std::endl;
    return Rcpp::List::create(
        Rcpp::Named("smoothed_samples") = smoothed_samples,
        Rcpp::Named("hyper_samples") = hyper_samples
    );
}