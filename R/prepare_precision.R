#' Prepare precision matrix components for GEV fitting
#'
#' @param parameters A vector of length 2 containing optimized parameters (rho1, rho2)
#' @param n_x Number of x-axis grid points
#' @param n_y Number of y-axis grid points
#' @param nu Smoothness parameter for the Matern covariance (default = 2)
#'
#' @return A list containing:
#'   \item{index}{Row indices of non-zero elements in L}
#'   \item{n_values}{Number of non-zero elements per column in L}
#'   \item{values}{Non-zero values in L}
#'   \item{log_det}{Log determinant of L}
#'
#' @export
prepare_precision <- function(rho, n_x, n_y, nu = 2) {
  # Extract parameters
  rho1 <- rho[1]
  rho2 <- rho[2]

  # Compute precision matrix and its Cholesky decomposition
  Q <- stdmatern::make_standardized_matern_eigen(n_x, n_y, rho1, rho2, nu)
  L <- Matrix::t(Matrix::chol(Q))

  # Extract sparse matrix information
  n_values <- Matrix::colSums(L != 0)
  index <- attributes(L)$i
  values <- attributes(L)$x
  log_det <- sum(log(Matrix::diag(L)))

  # Return components needed for GEV fitting
  list(
    index = index,
    n_values = n_values,
    values = values,
    log_det = log_det
  )
}
