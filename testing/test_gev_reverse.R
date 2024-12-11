library(tidyverse)
library(maxandsmooth)
library(Matrix)

model_stations <- bggjphd::stations |>
  filter(
    proj_x <= 40,
    proj_y <= 40
  )

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

str(res)


res$L

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

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = gamma)) +
  geom_raster()

plot_dat |>
  ggplot(aes(proj_x, proj_y, fill = delta)) +
  geom_raster()
