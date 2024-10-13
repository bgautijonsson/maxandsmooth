library(maxandsmooth)
library(bggjphd)
library(sf)
library(tidyverse)
library(evd)

model_stations <- stations |>
  filter(
    # proj_x <= 20,
    # proj_y <= 20
  )

model_precip <- precip |>
  semi_join(model_stations, by = "station")

Y <- model_precip |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()

tictoc::tic()
results <- max(Y, "gev")
tictoc::toc()


# exp(results[, 1]) |> range()
# exp(results[, 2] + results[, 1]) |> range()
# plogis(results[, 3]) |> range()

plot_dat <- tibble(
  psi = results$mles[, 1],
  tau = results$mles[, 2] + results$mles[, 1],
  phi = results$mles[, 3],
  mu = exp(psi),
  sigma = exp(tau + psi),
  xi = plogis(phi)
) |>
  mutate(
    station = model_stations$station,
    proj_x = model_stations$proj_x,
    proj_y = model_stations$proj_y
  ) |>
  inner_join(stations, by = c("station", "proj_x", "proj_y")) |>
  stations_to_sf() |>
  points_to_grid()

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
