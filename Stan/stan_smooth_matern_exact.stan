functions {
  /*
  Put an ICAR prior on coefficients
  */
  real icar_normal_lpdf(vector phi, real tau, int N, array[] int node1, array[] int node2) {
    return - 0.5 * tau * dot_self((phi[node1] - phi[node2])) +
      normal_lpdf(sum(phi) | 0, 0.001 * N);
  }

  real normal_prec_chol_lpdf(vector y, vector x, array[] int n_values, array[] int index, vector values, real log_det) {
    int N = num_elements(x);
    int counter = 1;
    vector[N] q = rep_vector(0, N);

    for (i in 1:N) {
      for (j in 1:n_values[i]) {
        q[i] += values[counter] * (y[index[counter]] - x[index[counter]]);
        counter += 1;
      }
    }

    return log_det - dot_self(q) / 2;
  }

  /*
    Computes the Kronecker product of two vectors a and b.

    @param a The first vector
    @param b The second vector

    @return A vector representing the Kronecker product of a and b
  */
  vector kronecker(vector a, vector b) {
    int n_a = rows(a);
    int n_b = rows(b);
    vector[n_a * n_b] result;
    
    for (i in 1:n_a) {
      for (j in 1:n_b) {
        result[(i-1)*n_b + j] = a[i] * b[j];
      }
    }
    
    return result;
  }

  /*
    Calculates the marginal standard deviations of the matrix Q, defined as the
    Kronecker sum of Q1 and Q2 with smoothness parameter nu.

    @param E1 Tuple containing the eigendecomposition of Q1
    @param E2 Tuple containing the eigendecomposition of Q2
    @param nu The smoothness parameter

    @return A vector of marginal standard deviations
  */
  vector marginal_sd(tuple(matrix, vector) E1, tuple(matrix, vector) E2, real nu) {
    int dim1 = cols(E1.1);
    int dim2 = cols(E2.1);

    matrix[dim1, dim1] V1 = E1.1;
    vector[dim1] A1 = E1.2;
    matrix[dim2, dim2] V2 = E2.1;
    vector[dim2] A2 = E2.2;

    vector[dim1 * dim2] marginal_sds = rep_vector(0.0, dim1 * dim2);
    for (i in 1:dim1) {
      for (j in 1:dim2) {
        vector[dim1 * dim2] v = kronecker(V1[, i], V2[, j]);
        real lambda = pow(A1[i] + A2[j], nu + 1);
        marginal_sds += square(v) / lambda;
      }
    }
    return sqrt(marginal_sds);
  }

  /*
    Computes the eigendecomposition of the precision matrix for an AR(1) process.

    @param n The size of the AR(1) process
    @param rho The correlation parameter

    @return A tuple containing the eigenvectors and eigenvalues of the precision matrix
  */
  tuple(matrix, vector) ar1_precision_eigen(int n, real rho) {
    matrix[n, n] Q;
    real scaling = 1.0 / (1.0 - rho * rho);
    real off_diag = -rho * scaling;

    Q = rep_matrix(0, n, n);
    for (i in 1:n) {
      Q[i, i] = (i == 1 || i == n) ? scaling : (1.0 + rho * rho) * scaling;
      if (i > 1) Q[i, i-1] = off_diag;
      if (i < n) Q[i, i+1] = off_diag;
    }
    
    return eigendecompose_sym(Q);
  }

  /*
    Computes the log probability density of a Matérn copula using the connection between 
    the eigendecomposition of the precision matrix and the smaller AR(1) precision matrices.

    @param Z The matrix of parameters
    @param dim1 The dimension of the first AR(1) process
    @param rho1 The correlation parameter of the first AR(1) process
    @param dim2 The dimension of the second AR(1) process
    @param rho2 The correlation parameter of the second AR(1) process
    @param nu The smoothness parameter

    @return The log probability density
  */
  real matern_exact_lpdf(matrix Z, int dim1, real rho1, int dim2, real rho2, int nu) {
    int D = dim1 * dim2;
    int N_param = cols(Z);
    tuple(matrix[dim1, dim1], vector[dim1]) E1 = ar1_precision_eigen(dim1, rho1);
    tuple(matrix[dim2, dim2], vector[dim2]) E2 = ar1_precision_eigen(dim2, rho2);


    real log_det = 0;
    real quadform_sum = 0;

    for (i in 1:dim1) {
      for (j in 1:dim2) {
        for (k in 1:N_param) {
          vector[D] v = kronecker(E1.1[, i], E2.1[, j]);
          
          real lambda = pow(E1.2[i] + E2.2[j], nu + 1);
          log_det += log(lambda);
          
          real q = v' * Z[, k];  
          quadform_sum += square(q) * lambda;
        }
      }
    }

    return -0.5 * (quadform_sum - log_det);
  }
}

data {
  int<lower = 1> n_stations;
  int<lower = 1> n_obs;
  int<lower = 1> n_param;

  int<lower = 1> dim1;
  int<lower = 1> dim2;
  int<lower = 0> nu;

  vector[n_stations * n_param] eta_hat;

  int<lower = 1> n_edges;
  array[n_edges] int<lower = 1, upper = n_stations> node1;
  array[n_edges] int<lower = 1, upper = n_stations> node2;

  int<lower = 1> n_nonzero_chol_Q;
  array[n_param * n_stations] int n_values;
  array[n_nonzero_chol_Q] int index;
  vector[n_nonzero_chol_Q] value;
  real<lower = 0> log_det_Q;
}

parameters {
  matrix[n_stations, 3] eta_raw;

  real mu_psi;
  real mu_tau;
  real mu_phi;

  vector[2] rho;    
}

model {
  vector[3 * n_stations] eta;
  eta[1:n_stations] = mu_psi + eta_raw[, 1];
  eta[(n_stations + 1):(2 * n_stations)] = mu_tau + eta_raw[, 2];
  eta[(2 * n_stations + 1):(3 * n_stations)] = mu_phi + eta_raw[, 3];


  target += normal_lpdf(eta_hat | eta, 1);
  target += matern_exact_lpdf(eta_raw | dim1, rho[1], dim2, rho[2], nu);
  target += std_normal_lpdf(rho);
}

