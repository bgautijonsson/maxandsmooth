#' Update station names in a data frame
#'
#' @param table Data frame to update
#' @param variable Variable containing station names
#' @param new_names Data frame with station to new_name mapping
#' @return Updated data frame
#' @importFrom dplyr inner_join mutate select
#' @keywords internal
update_names <- function(table, variable, new_names) {
  table |>
    inner_join(
      new_names,
      by = join_by({{ variable }} == station)
    ) |>
    mutate(
      "{{variable}}" := new_name
    ) |>
    select(-new_name)
}

#' Prepare precipitation data for Max-and-Smooth analysis
#'
#' @param stations Data frame containing station information
#' @param precip Data frame containing precipitation data
#' @param x_range Numeric vector of length 2 for x-axis range
#' @param y_range Numeric vector of length 2 for y-axis range
#'
#' @return A list containing:
#'   \item{Y}{Matrix of precipitation data}
#'   \item{stations}{Filtered stations data frame}
#'   \item{new_names}{Station name mapping}
#'   \item{edges}{Edge list for spatial neighborhood}
#'
#' @importFrom dplyr filter between semi_join mutate distinct select inner_join
#' @importFrom tidyr pivot_wider
#' @export
prepare_precip_data <- function(stations, precip,
                                x_range = c(0, 70),
                                y_range = c(46, 136)) {
  # Filter stations by location
  model_stations <- stations |>
    filter(
      between(proj_x, x_range[1], x_range[2]),
      between(proj_y, y_range[1], y_range[2])
    )

  # Create new station names
  new_names <- model_stations |>
    mutate(new_name = row_number()) |>
    distinct(station, new_name)

  # Filter and reshape precipitation data
  model_precip <- precip |>
    semi_join(model_stations, by = "station")

  Y <- model_precip |>
    pivot_wider(names_from = station, values_from = precip) |>
    select(-year) |>
    as.matrix()

  # Prepare edges with updated names
  edges <- bggjphd::twelve_neighbors |>
    filter(
      type %in% c("e", "n", "w", "s")
    ) |>
    inner_join(
      model_stations,
      by = join_by(station)
    ) |>
    semi_join(
      model_stations,
      by = join_by(neighbor == station)
    ) |>
    select(station, neighbor) |>
    update_names(station, new_names) |>
    update_names(neighbor, new_names)

  # Return prepared data
  list(
    Y = Y,
    stations = model_stations,
    new_names = new_names,
    edges = edges
  )
}
