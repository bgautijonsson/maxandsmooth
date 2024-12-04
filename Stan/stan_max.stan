functions {
  real gev_lpdf(vector y, real psi, real tau, real phi) {
    int n_obs = num_elements(y);

    real mu = exp(psi);
    real sigma = exp(tau + phi);
    real xi = inv_logit(phi);

    vector[n_obs] z = (y - mu) / sigma;
    real lp = 0;
    for (i in 1:n_obs) {
      if (abs(xi) < 1e-6) {
        lp += - log(sigma) - z[i] - exp(-z[i]);
      } else {
        real t = 1 + xi * z[i];
        lp += - log(sigma) - (1.0 + 1.0 / xi) * log(t) - pow(t, -1.0 / xi);
      }
    }

    return lp;
  }
}

data {
  int<lower = 1> n_stations;
  int<lower = 1> n_observations;
  matrix[n_observations, n_stations] y;
}

parameters {
  vector[n_stations] psi;
  vector[n_stations] tau;
  vector[n_stations] phi;
}

model {
  for (i in 1:n_stations) {
    target += gev_lpdf(y[, i] | psi[i], tau[i], phi[i]);
  }

  target += normal_lpdf(psi | 0, 1);
  target += normal_lpdf(tau | 0, 1);
  target += normal_lpdf(phi | 0, 1);
}
