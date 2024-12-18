#' Run the Max step of the Max-and-Smooth algorithm
#'
#' @param Y A matrix of precipitation data where rows are observations and columns are locations
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#'
#' @return A list containing:
#'   \item{parameters}{Estimated GEV parameters}
#'   \item{Hessian}{Hessian matrix of the fit}
#'   \item{L}{Cholesky factor of precision matrix}
#'   \item{stations}{Data frame with station information and parameter estimates}
#'
#' @importFrom tidyr pivot_wider
#' @importFrom dplyr mutate tibble inner_join
#' @export
ms_max <- function(Y, n_x, n_y, nu = 2) {
  # Step 1: Fit the Matern copula
  rho <- fit_copula(Y, n_x, n_y, nu)

  # Step 2: Prepare precision matrix components
  precision_components <- prepare_precision(
    rho,
    n_x,
    n_y,
    nu
  )

  # Step 3: Fit GEV distribution
  res <- fit_gev(
    Y,
    precision_components$index,
    precision_components$n_values,
    precision_components$values,
    precision_components$log_det
  )


  # Return results
  list(
    parameters = res$parameters,
    Hessian = res$Hessian,
    L = res$L,
    rho = rho
  )
}
