#' Run complete Max-and-Smooth analysis on precipitation data
#'
#' @param stations Data frame containing station information
#' @param precip Data frame containing precipitation data
#' @param x_range Numeric vector of length 2 for x-axis range
#' @param y_range Numeric vector of length 2 for y-axis range
#' @param ... Additional arguments passed to ms_smooth()
#'
#' @return A list containing results from both max and smooth steps
#' @export
maxandsmooth <- function(
    stations, precip,
    x_range = c(0, 70),
    y_range = c(46, 136),
    ...) {
  # Prepare data
  data <- prepare_precip_data(
    stations = stations,
    precip = precip,
    x_range = x_range,
    y_range = y_range
  )

  # Run Max step
  max_results <- ms_max(
    data = data$Y,
    n_x = length(unique(data$stations$proj_x)),
    n_y = length(unique(data$stations$proj_y))
  )

  # Run Smooth step
  smooth_results <- ms_smooth(
    max_step_results = max_results,
    stations = data$stations,
    edges = data$edges,
    ...
  )

  # Return results
  list(
    data = data,
    max_results = max_results,
    smooth_results = smooth_results
  )
}
