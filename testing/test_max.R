library(maxandsmooth)
library(bggjphd)
library(sf)
library(tidyverse)
library(evd)

model_stations <- stations

model_precip <- precip |>
  semi_join(model_stations, by = "station")

Y <- model_precip |>
  pivot_wider(names_from = station, values_from = precip) |>
  select(-year) |>
  as.matrix()

tictoc::tic()
results <- gev_mle_multiple(Y)
tictoc::toc()

exp(results[, 1]) |> range()
exp(results[, 2] + results[, 1]) |> range()
plogis(results[, 3]) |> range()
