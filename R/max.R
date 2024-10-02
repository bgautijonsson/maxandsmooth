#' Maximum Likelihood Estimation for GEV Distribution
#'
#' This function performs maximum likelihood estimation for the parameters of the Generalized Extreme Value (GEV) distribution.
#'
#' @param data A numeric vector of observations.
#' @return A list containing the estimated parameters: mu, sigma, and xi.
#' @export
gev_mle <- function(data) {
  .Call("_maxandsmooth_gev_mle", data)
}
