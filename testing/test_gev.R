library(maxandsmooth)
library(evd)
library(Matrix)

params <- c(7, 2, 0.1)

Y <- rgev(300, params[1], params[2], params[3])


res <- gev_mle(Y)

c(exp(res[1]), exp(res[2] + res[1]), plogis(res[3]))
