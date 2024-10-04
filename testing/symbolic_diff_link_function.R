library(Deriv)

# The functions in this file are the gradients and hessians of the GEV log-likelihood
# with respect to the transformed parameters
# psi = log(mu)
# tau = log(sigma) - log(mu)
# phi = logit(xi)
# With reverse link functions
# mu = exp(psi)
# sigma = exp(tau + psi)
# xi = inv_logit(phi)
# This means that mu and sigma are strictly positive, and xi is bounded between 0 and 1.

# GEV log-likelihood
gev_log_lik <- function(x, psi, tau, phi) {
  mu <- exp(psi)
  sigma <- exp(tau + psi)
  xi <- 1 / (1 + exp(-phi))


  z <- (x - mu) / sigma

  -log(sigma) - (1 + 1 / xi) * log(1 + xi * z) - (1 + xi * z)^(-1 / xi)
}

# Individual gradient components

# Gradient with respect to psi
Deriv(gev_log_lik, c("psi"))
grad_psi <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e7 <- (x - exp(psi)) / (.e3 * .e4) + 1
  -(1 + x * (1 / .e7^.e3 - (2 + .e2) / .e3) / (.e7 * .e4))
}

# Gradient with respect to tau
Deriv(gev_log_lik, c("tau"))
grad_tau <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e6 <- x - exp(psi)
  .e9 <- .e6 / (.e3 * .e4) + 1
  -((1 / .e9^.e3 - (2 + .e2) / .e3) * .e6 / (.e9 * .e4) + 1)
}

# Gradient with respect to phi
Deriv(gev_log_lik, c("phi"))
grad_phi <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e6 <- x - exp(psi)
  .e8 <- .e6 / (.e3 * .e4)
  .e9 <- .e8 + 1
  .e10 <- 2 + .e2
  -(((.e3 * (1 / .e9^.e3 - 1) * log1p(.e8) - .e6 / (.e9^.e10 *
    .e4)) * .e3 + .e10 * .e6 / (.e9 * .e4)) * .e2 / .e3^2)
}


# Hessian

# Psi Hessian
# Hessian with respect to psi, psi
Deriv(grad_psi, c("psi"))
hess_psi_psi <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e5 <- .e3 * .e4
  .e6 <- exp(psi)
  .e7 <- x - .e6
  .e9 <- .e7 / .e5 + 1
  .e10 <- .e5^2
  .e11 <- .e6 / .e5
  x * (((1 / .e5 - .e5 / .e10) * .e7 + 1 - .e11) * (1 / .e9^.e3 -
    (2 + .e2) / .e3) * .e4 / (.e9 * .e4)^2 - (.e5 * .e7 / .e10 +
    .e11) * .e3 / (.e9^(3 + .e2) * .e4))
}

# Hessian with respect to psi, tau
Deriv(grad_psi, c("tau"))
hess_psi_tau <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e5 <- .e3 * .e4
  .e7 <- x - exp(psi)
  .e9 <- .e7 / .e5 + 1
  .e10 <- .e5^2
  x * (((1 / .e5 - .e5 / .e10) * .e7 + 1) * (1 / .e9^.e3 - (2 + .e2) / .e3) *
    .e4 / (.e9 * .e4)^2 - .e3^2 * .e7 / (.e10 * .e9^(3 + .e2)))
}

# Hessian with respect to psi, phi
Deriv(grad_psi, c("phi"))
hess_psi_phi <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e5 <- .e3 * .e4
  .e7 <- x - exp(psi)
  .e8 <- .e7 / .e5
  .e9 <- .e8 + 1
  .e10 <- .e5^2
  .e11 <- .e9 * .e4
  .e12 <- .e9^.e3
  .e13 <- (2 + .e2) / .e3
  x * (((.e9^.e2 * .e3 * .e4 * .e7 / .e10 - .e12 * log1p(.e8)) / .e9^(2 *
    .e3) - (1 - .e13) / .e3) / .e11 + (1 / .e12 - .e13) * .e4^2 *
    .e7 / (.e11^2 * .e10)) * .e2
}

# Tau Hessian

# Hessian with respect to tau, tau
Deriv(grad_tau, c("tau"))
hess_tau_tau <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e5 <- .e3 * .e4
  .e7 <- x - exp(psi)
  .e9 <- .e7 / .e5 + 1
  .e10 <- .e5^2
  (((1 / .e5 - .e5 / .e10) * .e7 + 1) * (1 / .e9^.e3 - (2 + .e2) / .e3) *
    .e4 / (.e9 * .e4)^2 - .e3^2 * .e7 / (.e10 * .e9^(3 + .e2))) *
    .e7
}

# Hessian with respect to tau, phi
Deriv(grad_tau, c("phi"))
hess_tau_phi <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e6 <- x - exp(psi)
  .e7 <- .e3 * .e4
  .e8 <- .e6 / .e7
  .e9 <- .e8 + 1
  .e10 <- .e7^2
  .e11 <- .e9 * .e4
  .e12 <- .e9^.e3
  .e13 <- (2 + .e2) / .e3
  (((.e9^.e2 * .e3 * .e4 * .e6 / .e10 - .e12 * log1p(.e8)) / .e9^(2 *
    .e3) - (1 - .e13) / .e3) / .e11 + (1 / .e12 - .e13) * .e4^2 *
    .e6 / (.e11^2 * .e10)) * .e2 * .e6
}

# Phi Hessian
# Hessian with respect to phi, phi
Deriv(grad_phi, c("phi"))
hess_phi_phi <- function(x, psi, tau, phi) {
  .e2 <- exp(-phi)
  .e3 <- 1 + .e2
  .e4 <- exp(psi + tau)
  .e6 <- x - exp(psi)
  .e7 <- .e3 * .e4
  .e8 <- .e6 / .e7
  .e9 <- .e8 + 1
  .e10 <- .e9^.e3
  .e11 <- 2 + .e2
  .e12 <- .e7^2
  .e13 <- log1p(.e8)
  .e14 <- .e9 * .e4
  .e15 <- .e9^.e11
  .e17 <- 1 / .e10 - 1
  .e18 <- .e15 * .e4
  .e19 <- .e3 * .e17
  -((((((.e10 * .e11 * .e4 * .e6 / .e12 - .e15 * .e13) / .e18^2 +
    .e19 / (.e12 * .e9)) * .e4 * .e6 - (((.e9^.e2 * .e3 * .e4 *
    .e6 / .e12 - .e10 * .e13) * .e3 / .e9^(2 * .e3 - .e3) + 2) / .e10 -
    2) * .e13) * .e3 + (.e17 / .e14 - .e11 * .e4^2 * .e6 / (.e14^2 *
    .e12)) * .e6) * .e2 - ((.e19 * .e13 - .e6 / .e18) * .e3 +
    .e11 * .e6 / .e14) * (1 - 2 * (.e2 / .e3))) * .e2 / .e3^2)
}
