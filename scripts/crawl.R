library(crawl)
library(tidyverse)
library(plotly)

location <- read.csv(here::here("data/seal_locations.csv")) %>% 
  mutate(time = as.POSIXct(Date_es, 
                           format = "%Y-%m-%dT%H:%M:%SZ", 
                           tz = "UTC"), 
         lat = Latitude, 
         lon = Longitude)

depth <- read_csv(here::here("data/23A1302/out-Archive.csv"), show_col_types = FALSE) %>%
  drop_na(Depth) %>% 
  mutate(time = as.POSIXct(Time, format = "%H:%M:%S %d-%b-%Y", tz = "UTC")) %>% 
  rename(no_zoc_depth = Depth,
         Depth = `Corrected Depth`)

#fit the continuous time correlated random walk model

#i am not sure how to do the error, so just fitting 25 m in x/y directions for now
location$ln.sd.x <- log(25)  # 25m error in x direction
location$ln.sd.y <- log(25)  # 25m error in y direction

location <- location %>% distinct(time, .keep_all = TRUE)

coord_var <- var(location$lon, na.rm = TRUE) #get variance for initial state

initial_state <- list(
  a = c(location$lon[1], 0, location$lat[1], 0),
  P = diag(c(coord_var, coord_var * 10, coord_var, coord_var * 10))
)

fit <- crwMLE(
  mov.model = ~1,
  err.model = list(x = ~ln.sd.x - 1, y = ~ln.sd.y - 1),
  data = location,
  Time.name = "time",
  coord = c("lon", "lat"),
  initial.state = initial_state, 
  initialSANN = list(maxit = 1000), 
  attempts = 10, 
  control = list(maxit = 2000, trace = 1), 
  fixPar = c(1, 1, NA, NA)
)


#predict matching the depth data

valid_depth_times <- depth$time[
  depth$time >= min(location$time) & 
    depth$time <= max(location$time)
]

predictions <- crwPredict(
  object.crwFit = fit, 
  predTime = depth$time,
  return.type = "flat"
) 
predictions_df <- predictions %>% as.data.frame()

combined_data <- data.frame(
  time = predictions_df$time, 
  lon = predictions_df$mu.x, 
  lat = predictions_df$mu.y
)

combined_data <- merge(combined_data, depth, by = "time")

write_csv(combined_data, "output/h391_crawl.csv", append = FALSE)