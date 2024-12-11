library(cmdstanr)
library(tidyverse)
library(maxandsmooth)
library(Matrix)

model_stations <- bggjphd::stations |>
  filter(
    proj_x <= 3,
    proj_y <= 3
  )

model_precip <- bggjphd::precip |>
  semi_join(model_stations, by = "station")

Y <- model_precip |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()

stan_data <- list(
  N = nrow(Y),
  P = ncol(Y),
  Y = Y
)

model <- cmdstan_model(
  "Stan/stan_max.stan",
  force_recompile = TRUE
)
model$

fit <- model$optimize(
  data = stan_data,
  output_dir = "temp"
)

fit$init_model_methods(hessian = TRUE)




hess <- fit$hessian(
  unconstrained_variables = fit$summary(c("psi", "tau", "phi"))$estimate
)

hess$hessian |> image()
