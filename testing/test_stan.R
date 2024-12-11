library(maxandsmooth)
library(tidyverse)
library(cmdstanr)
library(here)
library(posterior)
library(Matrix)
library(bayesplot)

theme_set(
  bggjphd::theme_bggj()
)

update_names <- function(table, variable, new_names) {
  box::use(
    dplyr[inner_join, join_by, mutate, select]
  )
  table |>
    inner_join(
      new_names,
      by = join_by({{ variable }} == station)
    ) |>
    mutate(
      "{{variable}}" := new_name
    ) |>
    select(-new_name)
}

get_scaling_factor <- function(edges, N) {
  box::use(
    dplyr[filter, rename],
    Matrix[sparseMatrix, Diagonal, rowSums, diag],
    INLA[inla.qinv]
  )
  
  nbs <- edges |>
    filter(neighbor > station) |>
    rename(node1 = station, node2 = neighbor)
  
  adj.matrix <- sparseMatrix(i = nbs$node1, j = nbs$node2, x = 1, symmetric = TRUE)
  # The ICAR precision matrix (note! This is singular)
  Q <- Diagonal(N, rowSums(adj.matrix)) - adj.matrix
  # Add a small jitter to the diagonal for numerical stability (optional but recommended)
  Q_pert <- Q + Diagonal(N) * max(diag(Q)) * sqrt(.Machine$double.eps)
  
  # Compute the diagonal elements of the covariance matrix subject to the
  # constraint that the entries of the ICAR sum to zero.
  # See the inla.qinv function help for further details.
  Q_inv <- inla.qinv(Q_pert, constr = list(A = matrix(1, 1, N), e = 0))
  
  # Compute the geometric mean of the variances, which are on the diagonal of Q.inv
  scaling_factor <- exp(mean(log(diag(Q_inv))))
  
  scaling_factor
}


model_stations <- bggjphd::stations |>
  filter(
    between(proj_x, 0, 70),
    between(proj_y, 46, 136)
  )

new_names <- model_stations |>
  mutate(new_name = row_number()) |>
  distinct(station, new_name)

model_precip <- bggjphd::precip |>
  semi_join(model_stations, by = "station")

Y <- model_precip |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()

n_loc <- ncol(Y)
n_obs <- nrow(Y)

tictoc::tic()
res <- fit_gev(Y)
tictoc::toc()

#tictoc::tic()
#results <- ms_max(Y, "gev")
#tictoc::toc()

eta_hat <- res$parameters

#Q_pp <- Diagonal(x = results$Q_psi_psi)
#Q_tt <- Diagonal(x = results$Q_tau_tau)
#Q_ff <- Diagonal(x = results$Q_phi_phi)
#Q_pt <- Diagonal(x = results$Q_psi_tau)
#Q_pf <- Diagonal(x = results$Q_psi_phi)
#Q_tf <- Diagonal(x = results$Q_tau_phi)
#Q1 <- cbind(Q_pp, Q_pt, Q_pf)
#Q2 <- cbind(Q_pt, Q_tt, Q_tf)
#Q3 <- cbind(Q_pf, Q_tf, Q_ff)
#Q <- rbind(Q1, Q2, Q3)

L <- res$L

n_stations <- nrow(model_stations)



n_values <- colSums(L != 0)

index <- attributes(L)$i + 1

value <- attributes(L)$x

log_det_Q <- sum(log(diag(L)))

edges <- bggjphd::twelve_neighbors |>
  filter(
    type %in% c("e", "n", "w", "s")
  ) |>
  inner_join(
    model_stations,
    by = join_by(station)
  ) |>
  semi_join(
    model_stations,
    by = join_by(neighbor == station)
  ) |>
  select(station, neighbor) |>
  update_names(station, new_names) |>
  update_names(neighbor, new_names)

n_stations <- nrow(model_stations)
n_obs <- nrow(model_precip)
n_param <- 3

n_edges <- nrow(edges)
node1 <- edges$station
node2 <- edges$neighbor

dim1 <- length(unique(model_stations$proj_x))
dim2 <- length(unique(model_stations$proj_y))
nu <- 1

scaling_factor <- get_scaling_factor(edges, n_stations)

stan_data <- list(
  n_stations = n_stations,
  n_obs = n_obs,
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
eta_raw <- cbind(psi_raw/sd_psi, tau_raw/sd_tau, phi_raw/sd_phi)

inits <- list(
  psi = eta_hat[1:n_stations],
  tau = eta_hat[(n_stations + 1):(2 * n_stations)],
  phi = eta_hat[(2 * n_stations + 1):(3 * n_stations)],
  mu_psi = mean(psi_hat),
  mu_tau = mean(tau_hat),
  mu_phi = mean(phi_hat),
  eta_raw = eta_raw,
  mu = c(mu_psi, mu_tau, mu_phi),
  sigma = c(1, 1, 1),
  rho = c(0.5, 0.5, 0.5),
  eta_spatial = eta_raw,
  eta_random = matrix(0, nrow = nrow(eta_raw), ncol = ncol(eta_raw))
)

model <- cmdstan_model(
  here("Stan", "stan_smooth_bym2.stan")
)


fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  refresh = 100,
  iter_warmup = 1000,
  iter_sampling = 1000,
  init = list(inits, inits, inits, inits)
)

fit$time()

fit$summary(c("mu", "sigma", "rho"))

mcmc_trace(
  fit$draws("mu")
)

mcmc_trace(
  fit$draws("sigma"),
  transformations = log
)


mcmc_trace(
  fit$draws("rho"),
  transformations = \(x) log(x / (1 - x))
)


post_sum <- fit$summary("eta", mean)

plot_dat <- post_sum |> 
  filter(
    str_detect(variable, "^eta\\[")
  ) |> 
  mutate(
    station = str_match(variable, "eta\\[(.*),.*\\]")[, 2] |> parse_number(),
    variable = str_match(variable, "eta\\[.*,(.*)\\]")[, 2] |> parse_number(),
    variable = c("psi", "tau", "phi")[variable]
  ) |> 
    mutate(
      proj_x = model_stations$proj_x,
      proj_y = model_stations$proj_y,
      .by = variable
    )

uk <- bggjphd::get_uk_spatial(scale = "large")

d <- bggjphd::stations |> 
  bggjphd::stations_to_sf() |> 
  bggjphd::points_to_grid() |> 
  inner_join(
    plot_dat,
    by = join_by(proj_x, proj_y)
  )

d |> 
  filter(variable == "psi") |> 
  ggplot() +
  geom_sf(
    data = uk |> filter(name == "Ireland")
  ) +
  geom_sf(
    aes(fill = mean, col = mean),
    linewidth = 0.01,
    alpha = 0.6
) +
  scale_fill_viridis_c() +
  scale_colour_viridis_c()

d |> 
  filter(variable == "tau") |> 
  ggplot() +
  geom_sf(
    data = uk |> filter(name == "Ireland")
  ) +
  geom_sf(
    aes(fill = mean, col = mean),
    linewidth = 0.01,
    alpha = 0.6
  ) +
  scale_fill_viridis_c() +
  scale_colour_viridis_c()

d |> 
  filter(variable == "phi") |> 
  ggplot() +
  geom_sf(
    data = uk |> filter(name == "Ireland")
  ) +
  geom_sf(
    aes(fill = mean, col = mean),
    linewidth = 0.01,
    alpha = 0.6
  ) +
  scale_fill_viridis_c() +
  scale_colour_viridis_c()
