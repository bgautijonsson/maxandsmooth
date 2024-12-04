library(maxandsmooth)
library(bggjphd)
library(tidyverse)
library(Matrix)


model_stations <- stations |>
  filter(
    proj_x <= 10,
    proj_y <= 10
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


Q_pp <- Diagonal(x = results$Q_psi_psi)
Q_tt <- Diagonal(x = results$Q_tau_tau)
Q_ff <- Diagonal(x = results$Q_phi_phi)
Q_pt <- Diagonal(x = results$Q_psi_tau)
Q_pf <- Diagonal(x = results$Q_psi_phi)
Q_tf <- Diagonal(x = results$Q_tau_phi)

Q1 <- cbind(Q_pp, Q_pt, Q_pf)
Q2 <- cbind(Q_pt, Q_tt, Q_tf)
Q3 <- cbind(Q_pf, Q_tf, Q_ff)

Q <- rbind(Q1, Q2, Q3)
# Q |> image()

L <- chol(Q) |> t()
