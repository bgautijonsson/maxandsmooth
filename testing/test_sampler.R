library(maxandsmooth)
library(bggjphd)
library(tidyverse)
library(Matrix)
library(posterior)
library(bayesplot)


model_stations <- stations |>
  filter(
    proj_x < 25,
    proj_y < 25
  )

d <- precip |>
  semi_join(model_stations, by = join_by(station)) |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()

dim_x <- length(unique(model_stations$proj_x))
dim_y <- length(unique(model_stations$proj_y))

ms_res <- maxandsmooth(
  data = d,
  x_dim = dim_x,
  y_dim = dim_y,
  family = "gev",
  n_iterations = 300,
  burn_in = 100,
  thin = 1,
  print_every = 10,
  proposal_sd = 0.1
)

draws <- as_draws_df(ms_res)

draws

tibble(
  x = ms_res[, ncol(ms_res)]
) |>
  mutate(
    iter = row_number(),
    lagged = lag(x),
    accept = 1 * (x != lagged)
  ) |>
  drop_na() |>
  ggplot(aes(iter, accept)) +
  geom_smooth() +
  geom_point()

draws_summary <- summarise_draws(draws)
draws_summary

plot_dat <- draws_summary |>
  filter(str_detect(variable, "[0-9]")) |> 
  mutate(
    station = parse_number(variable),
    variable = str_replace(variable, "\\[.*", "")
  ) |> 
  mutate(
    proj_x = model_stations$proj_x,
    proj_y = model_stations$proj_y,
    .by = variable
  )

plot_dat |>
  filter(variable == "location") |>
  ggplot(aes(proj_x, proj_y, fill = mean)) +
  geom_raster()

plot_dat |>
  filter(variable == "scale") |>
  ggplot(aes(proj_x, proj_y, fill = mean)) +
  geom_raster()


plot_dat |>
  filter(variable == "shape") |>
  ggplot(aes(proj_x, proj_y, fill = mean)) +
  geom_raster()
