library(maxandsmooth)
library(sf)
library(tidyverse)
library(evd)

model_stations <- bggjphd::stations |>
  filter(
  )

model_precip <- bggjphd::precip |>
  semi_join(model_stations, by = "station")

Y <- model_precip |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()

tictoc::tic()
results <- ms_max(Y, "gevt")
tictoc::toc()



plot_dat <- tibble(
  value = results$eta_hat,
  name = rep(
    c("psi", "tau", "phi", "gamma"),
    each = nrow(model_stations)
  ),
  station = rep(
    seq_len(nrow(model_stations)),
    times = 4
  )
) |> 
  pivot_wider() |> 
  mutate(
    mu = exp(psi),
    sigma = exp(tau + psi),
    xi = plogis(phi),
    delta = 0.05 * plogis(gamma)
  ) |>
  mutate(
    station = model_stations$station,
    proj_x = model_stations$proj_x,
    proj_y = model_stations$proj_y
  ) |>
  inner_join(bggjphd::stations, by = c("station", "proj_x", "proj_y")) |>
  bggjphd::stations_to_sf() |>
  bggjphd::points_to_grid()

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
