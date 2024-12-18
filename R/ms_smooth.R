#' Run the Smooth step of the Max-and-Smooth algorithm using Stan
#'
#' @param max_step_results Results from ms_max() function
#' @param stations Data frame containing station information (must have proj_x and proj_y columns)
#' @param edges Data frame containing neighborhood structure (must have station and neighbor columns)
#' @param nu Smoothness parameter for the Matern covariance (default = 1)
#' @param chains Number of MCMC chains (default = 4)
#' @param iter_warmup Number of warmup iterations (default = 1000)
#' @param iter_sampling Number of sampling iterations (default = 1000)
#' @param parallel_chains Number of chains to run in parallel (default = 4)
#'
#' @return A cmdstanr model fit object
#'
#' @importFrom cmdstanr cmdstan_model
#' @importFrom dplyr filter select mutate
#' @importFrom Matrix sparseMatrix Diagonal rowSums diag
#' @export
ms_smooth <- function(
    max_step_results,
    stations,
    edges,
    nu = 1,
    chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    parallel_chains = 4) {
  # Extract dimensions and parameters
  n_stations <- nrow(stations)
  n_param <- 3
  eta_hat <- max_step_results$parameters
  L <- max_step_results$L

  # Process edges
  n_edges <- nrow(edges)
  node1 <- edges$station
  node2 <- edges$neighbor

  # Get grid dimensions
  dim1 <- length(unique(stations$proj_x))
  dim2 <- length(unique(stations$proj_y))

  # Process Cholesky components
  n_values <- colSums(L != 0)
  index <- attributes(L)$i + 1
  value <- attributes(L)$x
  log_det_Q <- sum(log(diag(L)))

  # Calculate scaling factor for BYM2 model
  scaling_factor <- get_scaling_factor(edges, n_stations)

  # Prepare Stan data
  stan_data <- list(
    n_stations = n_stations,
    n_param = n_param,
    eta_hat = eta_hat,
    n_edges = n_edges,
    node1 = node1,
    node2 = node2,
    n_nonzero_chol_Q = sum(n_values),
    n_values = n_values,
    index = index,
    value = value,
    log_det_Q = log_det_Q,
    dim1 = dim1,
    dim2 = dim2,
    nu = nu,
    scaling_factor = scaling_factor
  )

  # Prepare initial values
  psi_hat <- eta_hat[1:n_stations]
  tau_hat <- eta_hat[(n_stations + 1):(2 * n_stations)]
  phi_hat <- eta_hat[(2 * n_stations + 1):(3 * n_stations)]

  mu_psi <- mean(psi_hat)
  mu_tau <- mean(tau_hat)
  mu_phi <- mean(phi_hat)

  sd_psi <- sd(psi_hat)
  sd_tau <- sd(tau_hat)
  sd_phi <- sd(phi_hat)

  psi_raw <- psi_hat - mu_psi
  tau_raw <- tau_hat - mu_tau
  phi_raw <- phi_hat - mu_phi

  eta_raw <- cbind(psi_raw / sd_psi, tau_raw / sd_tau, phi_raw / sd_phi)

  inits <- list(
    psi = psi_hat,
    tau = tau_hat,
    phi = phi_hat,
    mu_psi = mu_psi,
    mu_tau = mu_tau,
    mu_phi = mu_phi,
    eta_raw = eta_raw,
    mu = c(mu_psi, mu_tau, mu_phi),
    sigma = c(1, 1, 1),
    rho = c(0.5, 0.5, 0.5),
    eta_spatial = eta_raw,
    eta_random = matrix(0, nrow = nrow(eta_raw), ncol = ncol(eta_raw))
  )

  # Compile and run Stan model
  model <- cmdstan_model(
    system.file("stan", "stan_smooth_bym2.stan", package = "maxandsmooth")
  )

  fit <- model$sample(
    data = stan_data,
    chains = chains,
    parallel_chains = parallel_chains,
    refresh = 100,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    init = rep(list(inits), chains)
  )

  return(fit)
}

#' Helper function to calculate scaling factor for BYM2 model
#'
#' @param edges Data frame containing neighborhood structure
#' @param N Number of stations
#' @return Scaling factor for BYM2 model
#' @importFrom Matrix sparseMatrix Diagonal rowSums diag
#' @importFrom INLA inla.qinv
get_scaling_factor <- function(edges, N) {
  # Filter and rename edges
  nbs <- edges |>
    filter(neighbor > station) |>
    rename(node1 = station, node2 = neighbor)

  # Create adjacency matrix
  adj.matrix <- sparseMatrix(
    i = nbs$node1,
    j = nbs$node2,
    x = 1,
    symmetric = TRUE
  )

  # Create ICAR precision matrix
  Q <- Diagonal(N, rowSums(adj.matrix)) - adj.matrix

  # Add small jitter for numerical stability
  Q_pert <- Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)

  # Compute inverse with sum-to-zero constraint
  Q_inv <- inla.qinv(
    Q_pert,
    constr = list(A = matrix(1, 1, N), e = 0)
  )

  # Return geometric mean of variances
  exp(mean(log(diag(Q_inv))))
}
