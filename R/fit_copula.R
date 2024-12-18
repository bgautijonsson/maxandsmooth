#' Fit a Matern copula to precipitation data
#'
#' @param data A matrix of precipitation data where rows are observations and columns are locations
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#' @param init Initial values for optimization (default = c(0.2, 0.4))
#'
#' @return A list containing:
#'   \item{parameters}{Optimized parameters (rho1, rho2)}
#'
#' @importFrom stats optim
#' @export
fit_copula <- function(Y, n_x, n_y, nu = 2, init = c(0.1, 0.1)) {
  # Optimize the Matern copula parameters
  par <- optim(
    init,
    fn = function(par) {
      rho1 <- par[1]
      rho2 <- par[2]
      -sum(stdmatern::dmatern_copula_eigen(Y, n_x, n_y, rho1, rho2, nu))
    }
  )$par

  # Return the optimized parameters
  return(par)
}
