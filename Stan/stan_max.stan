data {
  int<lower=0> N;  // number of observations per location
  int<lower=0> P;  // number of locations
  matrix[N, P] Y;  // data matrix: N observations × P locations
}

parameters {
  vector[P] psi;    // location parameters (transformed)
  vector[P] tau;    // scale parameters (transformed)
  vector[P] phi;    // shape parameters (transformed)
}

transformed parameters {
  vector[P] mu = exp(psi);
  vector[P] sigma = exp(psi + tau);
  vector[P] xi = inv_logit(phi);  // bounds xi between 0 and 1
}

model {
  // Priors
  psi ~ normal(0, 1);
  tau ~ normal(0, 1);
  phi ~ normal(0, 1);
  
  // Likelihood
  for (p in 1:P) {
    for (n in 1:N) {
      real z = (Y[n,p] - mu[p]) / sigma[p];
      
      if (xi[p] < 1e-6) {
        target += -log(sigma[p]) - z - exp(-z);
      } else {
        real t = 1 + xi[p] * z;
        target += -log(sigma[p]) - (1 + 1/xi[p]) * log(t) - pow(t, -1/xi[p]);
      }
    }
  }
}
