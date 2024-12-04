functions {
  /*
  Folds a vector representing a 2D grid into a larger vector with symmetric repetitions.
  This function is used to create a folded version of the data for the circulant approximation
  of a Matérn covariance.

  @param x The input vector representing the original 2D grid data
  @param n1 The first dimension of the original grid
  @param n2 The second dimension of the original grid

  @return A vector of length 4 * n1 * n2 containing the folded data
  */
  vector fold_data(vector x, int n1, int n2) {
    vector[4 * n1 * n2] folded;
    for (i in 1:n1) {
      for (j in 1:n2) {
        int idx = (i - 1) * n2 + j;
        folded[(i - 1) * 2 * n2 + j] = x[idx];
        folded[(i - 1) * 2 * n2 + (2 * n2 - j + 1)] = x[idx];
        folded[(2 * n1 - i) * 2 * n2 + j] = x[idx];
        folded[(2 * n1 - i) * 2 * n2 + (2 * n2 - j + 1)] = x[idx];
      }
    }
    return folded;
  }
  /*
  Creates a base matrix for the circulant approximation of a Matérn covariance
  defined as a Kronecker sum of two AR(1) processes approximated with 
  circulant matrices.

  @param dim1 The first dimension of the grid
  @param dim2 The second dimension of the grid
  @param rho1 The correlation parameter for the first dimension
  @param rho2 The correlation parameter for the second dimension

  @return A matrix representing the base for the circulant approximation
  */
  matrix make_base_matrix(int dim1, int dim2, real rho1, real rho2) {
    matrix[dim2, dim1] c = rep_matrix(0, dim2, dim1);
    vector[dim1] c1 = rep_vector(0, dim1);
    vector[dim2] c2 = rep_vector(0, dim2);

    real scale1 = 1.0 / (1.0 - square(rho1));  
    real scale2 = 1.0 / (1.0 - square(rho2));

    c1[1] = 1 + square(rho1);
    c1[2] = -rho1;
    c1[dim1] = -rho1;
    c1 *= scale1;

    c2[1] = 1 + square(rho2);
    c2[2] = -rho2;
    c2[dim2] = -rho2;
    c2 *= scale2;

    // Set the first row
    c[1, 1] = c1[1] + c2[1];  
    c[1, 2] = c1[2];
    c[1, dim1] = c1[dim1];

    // Set the rest of the first column
    c[2, 1] = c2[2];
    c[dim2, 1] = c2[dim2];
    return c;
  }

  /*
  Creates a base matrix and rescales its eigenvalues for the circulant approximation.
  The eigenvalues are rescaled so that the inverse of the precision matrix is
  a correlation matrix.

  The function returns the square root of the eigenvalues for use in quadratic forms.

  @param dim1 The first dimension of the grid
  @param dim2 The second dimension of the grid
  @param rho1 The correlation parameter for the first dimension
  @param rho2 The correlation parameter for the second dimension
  @param nu The smoothness parameter of the Matérn covariance

  @return A complex matrix of square roots of rescaled eigenvalues
  */
  complex_matrix create_base_matrix_and_rescale_eigenvalues(int dim1, int dim2, real rho1, real rho2, int nu) {
    
    matrix[dim2, dim1] c = make_base_matrix(dim1, dim2, rho1, rho2);

    // Compute the eigenvalues and marginal standard deviation
    complex_matrix[dim2, dim1] eigs = pow(fft2(c), (nu + 1.0));
    // complex_matrix[dim2, dim1] inv_eigs = pow(eigs, -1);
    // real mvar = get_real(inv_fft2(inv_eigs)[1, 1]);
    // eigs *= mvar;
    
    return eigs;
  }

  /*
  Performs matrix-vector multiplication in the Fourier domain.

  @param sqrt_eigs The square root of the eigenvalues
  @param v The vector to multiply

  @return The result of the matrix-vector product
  */
  vector matvec_prod(complex_matrix eigs, vector v) {
    int dim2 = rows(eigs);
    int dim1 = cols(eigs);
    complex_matrix[dim2, dim1] v_mat = to_matrix(v, dim2, dim1);
    
    complex_matrix[dim2, dim1] fft_v = fft2(v_mat);
    complex_matrix[dim2, dim1] prod = eigs .* fft_v;
    complex_matrix[dim2, dim1] result_complex = inv_fft2(prod);
    
    return to_vector(get_real(result_complex));
  }

  /*
  Computes the log probability density of a Matérn copula that's approximated
  with a block circulant matrix with circulant blocks.

  @param Z The vector of standard normal variates
  @param dim1 The first dimension of the grid
  @param dim2 The second dimension of the grid
  @param rho1 The correlation parameter for the first dimension
  @param rho2 The correlation parameter for the second dimension
  @param nu The smoothness parameter of the Matérn covariance

  @return The log probability density
  */
  real matern_folded_copula_lpdf(matrix Z, int dim1, int dim2, real rho1, real rho2, int nu) {
    complex_matrix[2 * dim2, 2 * dim1] eigs = create_base_matrix_and_rescale_eigenvalues(2 * dim1, 2 * dim2, rho1, rho2, nu);
    real quad_forms = 0;
    int N_params = cols(Z);
    real log_det = sum(log(get_real(eigs)));
    for (i in 1:N_params) {
      vector[4 * dim1 * dim2] Z_fold = fold_data(Z[, i], dim1, dim2);
      vector[4 * dim1 * dim2] Qz = matvec_prod(eigs, Z_fold);
      quad_forms += dot_product(Z_fold, Qz) - dot_self(Z_fold);
    }
    return - 0.5 * (quad_forms - log_det);
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
  

  int<lower = 1> n_nonzero_chol_Q;
  array[n_param * n_stations] int n_values;
  array[n_nonzero_chol_Q] int index;
  vector[n_nonzero_chol_Q] value;
  real<lower = 0> log_det_Q;
}

parameters {
  matrix[n_stations, n_param] eta_raw;

  real mu_psi;
  real mu_tau;
  real mu_phi;

  vector[2] rho;
}


model {
  vector[n_param * n_stations] eta;
  eta[1:n_stations] = mu_psi + eta_raw[, 1];
  eta[(n_stations + 1):(2 * n_stations)] = mu_tau + eta_raw[, 2];
  eta[(2 * n_stations + 1):(3 * n_stations)] = mu_phi + eta_raw[, 3];

  target += matern_folded_copula_lpdf(eta_raw | dim1, dim2, rho[1], rho[2], nu) / 4;

  target += normal_lpdf(eta_hat | eta, 1);
}

