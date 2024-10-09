#' Max-and-Smooth algorithm
#'
#' @param data A matrix of data, where each column represents a station and each row a time point
#' @param x_dim The x dimension of the spatial grid
#' @param y_dim The y dimension of the spatial grid
#' @param family The distribution family (currently only "gev" is supported)
#' @param n_iterations The number of MCMC iterations
#' @param burn_in The number of burn-in iterations
#' @param thin The thinning interval
#' @param print_every How often to print progress
#'
#' @return A list containing the smoothed samples and hyperparameter samples
#' @export
maxandsmooth <- function(
    data,
    x_dim,
    y_dim,
    family = "gev",
    n_iterations = 1000,
    burn_in = 1000,
    thin = 10,
    print_every = 100,
    proposal_sd = 0.1) {
  result <- maxandsmooth_cpp(
    data,
    x_dim,
    y_dim,
    family,
    n_iterations,
    burn_in,
    thin,
    print_every,
    proposal_sd
  )

  n_stations <- ncol(data)
  n_params <- 3 # For GEV distribution

  # Set column names for smoothed_samples
  smoothed_col_names <- c(
    paste0("location[", 1:n_stations, "]"),
    paste0("scale[", 1:n_stations, "]"),
    paste0("shape[", 1:n_stations, "]")
  )
  colnames(result$smoothed_samples) <- smoothed_col_names

  # Set column names for hyper_samples
  hyper_col_names <- c("tau_location", "tau_scale", "tau_shape")
  colnames(result$hyper_samples) <- hyper_col_names

  out <- cbind(
    smoothed_samples = result$smoothed_samples,
    hyper_samples = result$hyper_samples
  )

  return(out)
}
