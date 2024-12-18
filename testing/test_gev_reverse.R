library(tidyverse)
library(maxandsmooth)
library(stdmatern)
library(Matrix)

n_x <- 10
n_y <- 10

model_stations <- bggjphd::stations |>
  filter(
    proj_x <= n_x,
    proj_y <= n_y
  )

model_precip <- bggjphd::precip |>
  semi_join(model_stations, by = "station")

Y <- model_precip |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()


n_loc <- ncol(Y)
n_obs <- nrow(Y)
nu <- 2

par <- optim(
  c(0.2, 0.4),
  fn = function(par) {
    rho1 <- par[1]
    rho2 <- par[2]
    -dmatern_copula_eigen(Y, n_x, n_y, rho1, rho2, nu) |> sum()
  }
)$par 


Q <- make_standardized_matern_eigen(n_x, n_y, par[1], par[2], nu)
L <- chol(Q) |> t()
n_values <- colSums(L != 0)
index <- attributes(L)$i
value <- attributes(L)$x
log_det <- sum(log(diag(L)))

tictoc::tic()
res <- fit_gev(Y, index, n_values, value, log_det)
tictoc::toc()

str(res)

-res$Hessian |> image()
-res$Hessian |> solve() |> cov2cor()
colSums(res$Hessian != 0)
rowSums(res$Hessian != 0)

mean(res$Hessian != 0)

res$L |> image()

d <- tibble(
  value = res$parameters,
  name = rep(
    c("psi", "tau", "phi"),
    each = n_loc
  ),
  station = rep(
    seq_len(n_loc),
    times = 3
  )
) |>
  pivot_wider() |>
  mutate(
    mu = exp(psi),
    sigma = exp(tau + psi),
    xi = plogis(phi)
  ) |>
  mutate(
    station = model_stations$station,
    proj_x = model_stations$proj_x,
    proj_y = model_stations$proj_y
  ) |>
  inner_join(bggjphd::stations, by = c("station", "proj_x", "proj_y"))

plot_dat <- d |>
  bggjphd::stations_to_sf() |>
  bggjphd::points_to_grid(
    n_x = length(unique(d$proj_x)),
    n_y = length(unique(d$proj_y))
  )


plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = psi)) +
  geom_raster()

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = mu)) +
  geom_raster()

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = tau)) +
  geom_raster()

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = sigma)) +
  geom_raster()

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = phi)) +
  geom_raster()

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = xi)) +
  geom_raster()

