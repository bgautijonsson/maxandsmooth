library(Deriv)

# GEV log-likelihood
gev_log_lik <- function(x, mu, sigma, xi) {
  z <- (x - mu) / sigma

  -log(sigma) - (1 + 1 / xi) * log(1 + xi * z) - (1 + xi * z)^(-1 / xi)
}

# Individual gradient components

# Gradient with respect to mu
Deriv(gev_log_lik, c("mu"))
grad_mu <- function(x, mu, sigma, xi) {
  .e1 <- 1 + xi * (x - mu) / sigma
  .e2 <- 1 / xi
  -((1 / .e1^.e2 - xi * (1 + .e2)) / (sigma * .e1))
}

# Gradient with respect to sigma
Deriv(gev_log_lik, c("sigma"))
grad_sigma <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e2 <- 1 + xi * .e1 / sigma
  .e3 <- 1 / xi
  -(((1 / .e2^.e3 - xi * (1 + .e3)) * .e1 / (sigma * .e2) + 1) / sigma)
}

# Gradient with respect to xi
Deriv(gev_log_lik, c("xi"))
grad_xi <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e3 <- xi * .e1 / sigma
  .e4 <- 1 + .e3
  .e5 <- 1 / xi
  .e6 <- 1 + .e5
  -(((1 / .e4^.e5 - 1) * log1p(.e3) / xi - .e1 / (sigma * .e4^.e6)) / xi +
    .e6 * .e1 / (sigma * .e4))
}

# Hessian

# Mu Hessian
# Hessian with respect to mu, mu
Deriv(grad_mu, c("mu", "mu"))
grad_mu_mu <- function(x, mu, sigma, xi) {
  .e1 <- 1 + xi * (x - mu) / sigma
  .e2 <- 1 / xi
  -(xi * (1 / (sigma * xi * .e1^(.e2 + 2)) + sigma * (1 / .e1^.e2 - xi * (1 + .e2)) / (sigma * .e1)^2) / sigma)
}

# Hessian with respect to mu, sigma
Deriv(grad_mu, c("sigma"))
grad_mu_sigma <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e2 <- 1 + xi * .e1 / sigma
  .e3 <- 1 / xi
  (1 / .e2^.e3 - xi * (1 + .e3)) / (sigma * .e2)^2 - .e1 / (sigma^3 * .e2^(.e3 + 2))
}

# Hessian with respect to mu, xi
Deriv(grad_mu, c("xi"))
grad_mu_xi <- function(x, mu, sigma, xi) {
  e1 <- x - mu
  .e3 <- xi * .e1 / sigma
  .e4 <- 1 + .e3
  .e5 <- 1 / xi
  .e6 <- .e4^.e5
  .e7 <- sigma * .e4
  ((.e4^(.e5 - 1) * .e1 / sigma - .e6 * log1p(.e3) / xi) / (xi * .e4^(2 / xi)) + 1) / .e7 + (1 / .e6 - xi * (1 + .e5)) * .e1 / .e7^2
}

# Sigma Hessian

# Hessian with respect to sigma, mu
Deriv(grad_sigma, c("mu"))
grad_sigma_mu <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e2 <- 1 + xi * .e1 / sigma
  .e3 <- 1 / xi
  .e4 <- 1 + .e3
  .e6 <- 1 / .e2^.e3
  .e7 <- sigma * .e2
  .e8 <- xi * .e4
  ((.e6 - (.e1 / (sigma * .e2^.e4) + .e8)) / .e7 - xi * (.e6 - .e8) * .e1 / .e7^2) / sigma
}

# Hessian with respect to sigma, sigma
Deriv(grad_sigma, c("sigma"))
grad_sigma_sigma <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e2 <- 1 + xi * .e1 / sigma
  .e3 <- 1 / xi
  .e7 <- 1 / .e2^.e3 - xi * (1 + .e3)
  .e8 <- sigma * .e2
  ((.e7 * .e1 / .e8 + 1) / sigma + (.e7 / .e8^2 - .e1 / (sigma^3 * .e2^(.e3 + 2))) * .e1) / sigma
}

# Hessian with respect to sigma, xi
Deriv(grad_sigma, c("xi"))
grad_sigma_xi <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e3 <- xi * .e1 / sigma
  .e4 <- 1 + .e3
  .e5 <- 1 / xi
  .e6 <- .e4^.e5
  .e7 <- sigma * .e4
  (((.e4^(.e5 - 1) * .e1 / sigma - .e6 * log1p(.e3) / xi) / (xi * .e4^(2 / xi)) + 1) / .e7 + (1 / .e6 - xi * (1 + .e5)) * .e1 / .e7^2) * .e1 / sigma
}

# Xi Hessian
# Hessian with respect to xi, xi
Deriv(grad_xi, c("xi"))
grad_xi_xi <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e3 <- xi * .e1 / sigma
  .e4 <- 1 + .e3
  .e5 <- 1 / xi
  .e6 <- 1 + .e5
  .e7 <- log1p(.e3)
  .e8 <- .e4^.e6
  .e9 <- .e4^.e5
  .e10 <- sigma * .e4
  .e11 <- sigma * .e8
  .e12 <- xi^2
  -((((.e1 / .e10 - 2 * (.e7 / xi)) * (1 / .e9 - 1) + .e1 / .e11 -
    (.e4^(.e5 - 1) * .e1 / sigma - .e9 * .e7 / xi) * .e7 / (xi *
      .e4^(2 / xi))) / xi + sigma * (.e6 * .e9 * .e1 / sigma -
    .e8 * .e7 / .e12) * .e1 / .e11^2) / xi - (.e6 * .e1 / .e10^2 +
    1 / (sigma * .e12 * .e4)) * .e1)
}

# Hessian with respect to xi, mu
Deriv(grad_xi, c("mu"))
grad_xi_mu <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e2 <- xi * .e1
  .e3 <- .e2 / sigma
  .e4 <- 1 + .e3
  .e5 <- 1 / xi
  .e6 <- 1 + .e5
  .e7 <- .e4^.e6
  .e8 <- .e4^.e5
  .e9 <- sigma * .e4
  -((1 / .e7 - xi * (((1 - log1p(.e3) / xi) / .e8 - 1) / (xi * .e4) +
    sigma * .e6 * .e8 * .e1 / (sigma * .e7)^2)) / (sigma * xi) -
    .e6 * (1 / .e9 - .e2 / .e9^2))
}

# Hessian with respect to xi, sigma
Deriv(grad_xi, c("sigma"))
grad_xi_sigma <- function(x, mu, sigma, xi) {
  .e1 <- x - mu
  .e3 <- xi * .e1 / sigma
  .e4 <- 1 + .e3
  .e5 <- 1 / xi
  .e6 <- 1 + .e5
  .e7 <- .e4^.e6
  .e8 <- .e4^.e5
  -((((.e7 - xi * .e6 * .e8 * .e1 / sigma) / (sigma * .e7)^2 -
    ((1 - log1p(.e3) / xi) / .e8 - 1) / (sigma^2 * .e4)) / xi - .e6 / (sigma *
    .e4)^2) * .e1)
}
