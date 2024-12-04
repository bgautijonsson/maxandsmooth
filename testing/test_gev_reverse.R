library(maxandsmooth)
library(evd)

params <- c(7, 2, 0.1)

n_obs <- 30
n_loc <- 1000

Y <- matrix(
  0,
  nrow = n_obs,
  ncol = n_loc
)

for (j in 1:n_loc) {
  Y[ , j] <- rgev(n_obs, params[1], params[2], params[3])
}

bench::mark(
  gradient(Y)
)
