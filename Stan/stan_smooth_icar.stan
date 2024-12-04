functions {
  /*
  Put an ICAR prior on coefficients
  */
  real icar_normal_lpdf(vector phi, int N, array[] int node1, array[] int node2) {
    return - 0.5 * dot_self((phi[node1] - phi[node2])) +
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
}

data {
  int<lower = 1> n_stations;
  int<lower = 1> n_obs;
  int<lower = 1> n_param;

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

transformed data {
  vector[n_stations] psi_hat = eta_hat[1:n_stations];
  vector[n_stations] tau_hat = eta_hat[(n_stations + 1):(2 * n_stations)];
  vector[n_stations] phi_hat = eta_hat[(2 * n_stations + 1):(3 * n_stations)];
}

parameters {
  vector[n_stations] psi;
  vector[n_stations] tau;
  vector[n_stations] phi;

  real mu_psi;
  real mu_tau;
  real mu_phi;
}


  

model {
  vector[n_param * n_stations] eta;
  eta[1:n_stations] = mu_psi + psi;
  eta[(n_stations + 1):(2 * n_stations)] = mu_tau + tau;
  eta[(2 * n_stations + 1):(3 * n_stations)] = mu_phi + phi;

  target += icar_normal_lpdf(psi | n_edges, node1, node2);
  target += icar_normal_lpdf(tau | n_edges, node1, node2);
  target += icar_normal_lpdf(phi | n_edges, node1, node2);

  target += normal_lpdf(eta_hat | eta, 1);
}

