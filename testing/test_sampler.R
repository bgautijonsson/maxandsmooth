library(maxandsmooth)
library(bggjphd)
library(tidyverse)
library(Matrix)
library(posterior)
library(bayesplot)


model_stations <- stations |>
  filter(
    between(proj_x, 100, 125),
    between(proj_y, 100, 125)
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
  n_iter = 500,
  n_burnin = 500,
  n_thin = 1,
  proposal_sd = 0.5
)

str(ms_res)

tau_post <- ms_res$smooth_results$tau
colnames(tau_post) <- c("tau_psi", "tau_tau", "tau_phi")
eta_post <- ms_res$smooth_results$eta
colnames(eta_post) <- c(
  paste0("psi[", seq_len(nrow(model_stations)), "]"),
  paste0("tau[", seq_len(nrow(model_stations)), "]"),
  paste0("phi[", seq_len(nrow(model_stations)), "]")
)

post <- cbind(
  tau_post,
  eta_post
)

draws <- as_draws_df(post)

draws

mcmc_trace(
  draws,
  pars = c("tau_psi", "tau_tau", "tau_phi")
)

tibble(
  x = draws[, 1]
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
  filter(variable == "psi") |>
  ggplot(aes(proj_x, proj_y, fill = mean)) +
  geom_raster()

plot_dat |>
  filter(variable == "tau") |>
  ggplot(aes(proj_x, proj_y, fill = mean)) +
  geom_raster()


plot_dat |>
  filter(variable == "phi") |>
  ggplot(aes(proj_x, proj_y, fill = mean)) +
  geom_raster()
