##########################################################
# Goal: Predict the rating a movie will receive
# (recommendation system)
##########################################################

####################### Step 1: Pre-processing / data exploration #######################
## Training set = edx_train
## Test set = edx_test
## Validation set = validation

# Install tidyverse, caret, lubridate, gridExtra, extrafont and data.table packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

if(!require(extrafont)) install.packages("extrafont", repos = "http://cran.us.r-project.org")
if(!require(extrafontdb)) install.packages("extrafontdb", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(lubridate)

library(data.table)
library(gridExtra)

library(extrafont)
library(extrafontdb)

# set significant digits to 5
options(digits = 5)

### PRE-PROCESSING
## Download additional fonts (with extrafont package)
font_import()
loadfonts()

## Download data
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# create a function that extracts the year a movie was released by taking a substring of the movie title
movie_to_year <- function(movie){
  temp <- substr(movie, nchar(movie)-4, nchar(movie)-1)
  as.numeric(temp)
}

# update movielens dataset
movielens <- movielens %>% mutate(movie_year = movie_to_year(title))

## Create validation/edx set
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Clear workspace
rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Create edx_train/edx_test sets
# Test set will be 20% of edx data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-test_index,]
temp_test <- edx[test_index,]

# Make sure userId and movieId in temp_test set are also in edx_train set
edx_test <- temp_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp_test, edx_test)
edx_train <- rbind(edx_train, removed)

# Clear workspace
rm(temp_test, removed, test_index)

## DATA EXPLORATION
str(edx_train)

# Get dimensions of edx_test and edx_train sets
dim_edx_test <- dim(edx_test)
dim_edx_train <- dim(edx_train)

# Confirm that test set contains 80% of the data
dim_edx_test[1]/sum(dim_edx_test[1], dim_edx_train[1])

# edx_train - all predictors
colnames(edx_train)

# edx_train - rating distribution
unique_ratings <- edx_train %>% group_by(rating) %>%
  summarise(n = n())

unique_ratings %>%
  ggplot(aes(rating, n)) +
  geom_bar(stat = "identity", col = "black") +
  xlab("Rating") +
  ylab("Number of Ratings") +
  ggtitle("Rating Distribution (in the training set)") +
  theme(text = element_text(family = "Times New Roman", size = 10))

# edx_train - movie_year distribution
edx_train %>% group_by(movie_year) %>%
  summarise(n = n(), avg_rating = mean(rating)) %>%
  ggplot(aes(x=movie_year, y=n)) +
  geom_bar(stat = "identity", col="black") +
  ylab("Number of Movies") +
  xlab("Movie Release Year") +
  ggtitle("Number of Movies Released vs Year of Release") +
  theme(text = element_text(family = "Times New Roman", size = 10))

# calculate number of unique users/movies/genres
edx_train %>% summarise(n_movies = n_distinct(movieId), n_users = n_distinct(userId), n_genres = n_distinct(genres))

# clear workspace
rm(dim_edx_test, dim_edx_train)

####################### Step 2: Data visualization #######################

## edx_train: genres
# find the number of movies in each genre (distinct)
movies <- edx_train %>% distinct(movieId, .keep_all = TRUE)
unique_genres_list <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                   "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                   "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX")
n_movies_per_genre <- sapply(unique_genres_list, function(g) {
  sum(str_detect(movies$genres, g))
})

# make genres and the number of movies per genre into a data frame
movies_per_genre <- data.frame(genres = unique_genres_list, n = n_movies_per_genre)

genre_plot <- movies_per_genre %>% ggplot(mapping = aes(reorder(genres, -n), n)) +
  geom_bar(stat = "identity") +
  xlab("Genre") +
  ylab("Number of Movies") +
  ggtitle("Number of Movies per Genre") +
  theme(axis.text.x = element_text(angle = 45)) +
  theme(text = element_text(family = "Times New Roman", size = 10))
genre_plot

## edx_train: userId (users)
unique_users <- edx_train %>% group_by(userId) %>%
  summarise(n = n(), avg_rating = mean(rating), rounded_rating = round(avg_rating*2)/2)

# number of users vs average rating
user_plot1 <- unique_users %>%
  ggplot(aes(rounded_rating)) +
  geom_histogram(bins = 10, col = "black") +
  xlab("Average Rating") +
  ylab("Number of Users") +
  ggtitle("Number of Users versus Average Movie Rating") +
  theme(text = element_text(family = "Times New Roman", size=10))

(unique_users %>% filter(rounded_rating == 3.5) %>% nrow())/nrow(unique_users) # most common rating
(unique_users %>% filter(rounded_rating < 2 | rounded_rating == 5) %>% nrow())/nrow(unique_users) # least common ratings

# number of ratings per user (distribution): most users have given between 500 and 2000 ratings
user_plot2 <- unique_users %>% filter(n > 500) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 20, col = "black") +
  xlab("Number of Ratings") +
  ggtitle("Number of Ratings per User") +
  theme(text = element_text(family = "Times New Roman", size=10))

(unique_users %>% filter(n < 2000) %>% nrow()) / nrow(unique_users) # % users with <2000 total ratings

# avg rating versus number of ratings given by user
user_plot3 <- unique_users %>%
  ggplot(aes(rounded_rating, n)) +
  geom_point(alpha = 0.5) +
  xlab("Average Rating (rounded)") +
  ylab("Number of Ratings") +
  ggtitle("Average Rating versus number of Ratings per User") +
  theme(text = element_text(family = "Times New Roman", size=10))

# commbine all user plots
grid.arrange(user_plot1, user_plot2, user_plot3)

## edx_train - movieId/title
unique_movies <- edx %>% group_by(movieId, title) %>%
  summarise(n = n(), avg_rating = mean(rating), rounded_rating = round(avg_rating*2)/2)

# number of movies vs average rating
movie_plot1 <- unique_movies %>%
  ggplot(aes(rounded_rating)) +
  geom_histogram(bins = 10, col = "black") +
  xlab("Average Rating") +
  ylab("Number of Movies") +
  ggtitle("Number of Movies versus Average Movie Rating") +
  theme(text = element_text(family = "Times New Roman", size=10))

movie_plot2 <- unique_movies %>% filter(n > 200) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 20, col = "black") +
  xlab("Number of Ratings") +
  ggtitle("Number of Ratings per Movie") +
  theme(text = element_text(family = "Times New Roman", size=10))

# avg rating versus number of ratings given for movie
movie_plot3 <- unique_movies %>%
  ggplot(aes(rounded_rating, n)) +
  geom_point(alpha = 0.5) +
  xlab("Rounded Rating") +
  ggtitle("Average Rating vs Number of Ratings per Movie") +
  theme(text = element_text(family = "Times New Roman", size=10))

# combine all movie plots
grid.arrange(movie_plot1, movie_plot2, movie_plot3)

## edx_train - movie_year, plot relationship between year a movie was released in and its rating
edx_train %>% group_by(movie_year) %>%
  summarise(n = n(), avg_rating = mean(rating)) %>%
  ggplot(aes(x=movie_year, y=avg_rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Movie Release Year") +
  ylab("Average Rating") +
  ggtitle("Average Rating vs Movie Release Year") +
  theme(text = element_text(family = "Times New Roman", size=10))

## edx_train - genres
unique_genres <- edx_train %>% group_by(genres) %>% 
  summarize(n_ratings = n(), avg_rating = mean(rating), se = sd(rating)/sqrt(n_ratings))

unique_genres %>%
  filter(n_ratings > 1000) %>%
  ggplot(data = .) +
  geom_bar(aes(x=genres, y=avg_rating), stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  geom_errorbar(aes(x=genres, ymin=avg_rating - 2*se, ymax = avg_rating + 2*se)) +
  xlab("Genre(s)") +
  ylab("Average Rating") +
  ggtitle("Rating Distribution per Genre") +
  theme(text = element_text(family = "Times New Roman", size = 10))

# Count the number of genres each movie has (only count unique movies)
n_genres <- str_count(movies$genres, pattern = "\\|") + 1
n_genres <- table(n_genres) # see how many genres each rated movie is defined with

# most movies have 0, 1, 2 or 3 different genres 
n_genres_percentage <- 100*n_genres/sum(n_genres) # find % of movies

# Get the first three genres of each movie
genre_columns <- paste("genre", 1:3, sep = "_")
no_genre <- "(no genre listed)"
t <- edx_train %>% separate(genres, sep = "\\|", into = genre_columns, extra = "drop", fill = "right")
t <- t %>% replace_na(list(genre_1 = no_genre, genre_2 = no_genre, genre_3 = no_genre))

# plot the rating distribution across genres for genre_1
genre_1_plot <- t %>% group_by(genre_1) %>% 
  summarise(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n)) %>%
  ggplot(data = .) +
  geom_bar(aes(x=genre_1, y=avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  geom_errorbar(aes(x=genre_1, ymin=avg - 2*se, ymax = avg + 2*se)) +
  xlab("First Genre") +
  ylab("Average Rating") +
  ggtitle("Ratings, Genre 1") +
  theme(text = element_text(family = "Times New Roman", size=10))

# plot the rating distribution across genres for genre_2
genre_2_plot <- t %>% group_by(genre_2) %>% 
  summarise(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n)) %>%
  ggplot(data = .) +
  geom_bar(aes(x=genre_2, y=avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  geom_errorbar(aes(x=genre_2, ymin=avg - 2*se, ymax = avg + 2*se)) +
  xlab("Second Genre") +
  ylab("Average Rating") +
  ggtitle("Rating, Genre 2") +
  theme(text = element_text(family = "Times New Roman", size=10))

# plot the rating distribution across genres for genre_3
genre_3_plot <- t %>% group_by(genre_3) %>% 
  summarise(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n)) %>%
  ggplot(data = .) +
  geom_bar(aes(x=genre_3, y=avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  geom_errorbar(aes(x=genre_3, ymin=avg - 2*se, ymax = avg + 2*se)) +
  xlab("Third Genre") +
  ylab("Average Rating") +
  ggtitle("Ratings, Genre 3") +
  theme(text = element_text(family = "Times New Roman", size=10))

# no regions of overlap in all three graphs -> statistically significant effect
grid.arrange(genre_1_plot, genre_2_plot, genre_3_plot, ncol=3)

## edx_train - timestamp
# with loess, the time a rating was made has an impact on the observed rating
dates <- edx_train %>% 
  mutate(date = as_datetime(timestamp)) %>%
  mutate(date = round_date(date, unit = "week")) # date in weeks

# plot week of rating vs rating given
dates %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Date of Rating") +
  ylab("Rating") +
  ggtitle("Date (Week) of Rating versus Rating Given") +
  theme(text = element_text(family = "Times New Roman", size=10))

# clear workspace
rm(genre_1_plot, genre_2_plot, genre_3_plot, movie_plot1, movie_plot2, movie_plot3,
   user_plot1, user_plot2, user_plot3)

####################### Step 3: Predictor Effects  #######################
# Predictors: userId, movieId/title, genres, timestamp

# average rating (across entire edx_train set)
mu <- mean(edx_train$rating)

## userId bias, b_u
# this plot shows that there is variation in the deviation of avg user ratings from the mean mu
# variability across users
user_bias <- edx_train %>% group_by(userId) %>% 
  summarise(b_u = sum(rating - mu)/n()) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 10, col = "black") +
  xlab("User Effect, b_u") +
  ggtitle("User Effect, b_u") +
  theme(text = element_text(family = "Times New Roman", size=10))

## movieId/title bias, b_i
# this plot shows a similar effect as userId
# variability across movies
movie_bias <- edx_train %>% group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/n()) %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 10, col = "black") +
  xlab("Movie Effect, b_i") +
  ggtitle("Movie Effect, b_i") +
  theme(text = element_text(family = "Times New Roman", size=10))

## movie_year bias, b_my
movie_year_bias <- edx_train %>% group_by(movie_year) %>%
  summarise(b_my = sum(rating - mu)/n()) %>%
  ggplot(aes(b_my)) +
  geom_histogram(bins = 10, col = "black") +
  xlab("Movie Year Effect, b_my") +
  ggtitle("Movie Year Effect, b_my") +
  theme(text = element_text(family = "Times New Roman", size=10))

## date/timsetamp bias, d_ui
unique_dates <- dates %>%
  group_by(date) %>%
  summarise(d_ui = sum(rating - mu)/n(), n = n(), avg_rating = mean(rating)) 

time_bias <- unique_dates %>%
  ggplot(aes(d_ui)) +
  geom_histogram(bins = 20, col = "black")+
  xlab("Date Effect, d_ui") +
  ggtitle("Date Effect, d_ui") +
  theme(text = element_text(family = "Times New Roman", size=10))

# combine and plot each predictor bias
grid.arrange(user_bias, movie_bias, time_bias, movie_year_bias)
rm(user_bias, movie_bias, time_bias, movie_year_bias)

## genre
# g_ui can be described as the sum of the effects across the first three genres
# note: put the equation for the confidence interval into the report

# genres
genres_bias <- edx_train %>% group_by(genres) %>%
  summarise(g_ui = sum(rating - mu)/ n()) %>%
  ggplot(aes(g_ui)) +
  geom_histogram(bins = 20, col = "black") +
  xlab("Genre Effect, g_ui") +
  ggtitle("Genre Effect")

# genre_1
g_ui1 <- t %>% group_by(genre_1) %>%
  summarise(g_ui = sum(rating - mu)/n()) 

genre1_bias <- g_ui1 %>%
  ggplot(aes(g_ui)) +
  geom_histogram(bins = 20, col = "black") +
  xlab("First Genre Effect, g_ui1") +
  ggtitle("First Genre Effect")

# genre_2
g_ui2 <- t %>% group_by(genre_2) %>%
  summarise(g_ui = sum(rating - mu)/n())

genre2_bias <- g_ui2 %>%
  ggplot(aes(g_ui)) +
  geom_histogram(bins = 20, col = "black") +
  xlab("Second Genre Effect, g_ui2") +
  ggtitle("Second Genre Effect")

# genre_3
g_ui3 <- t %>% group_by(genre_3) %>%
  summarise(g_ui = sum(rating - mu)/n())

genre3_bias <- g_ui3 %>%
  ggplot(aes(g_ui)) +
  geom_histogram(bins = 20, col = "black") +
  xlab("Third Genre Effect, g_ui3") +
  ggtitle("Third Genre Effect")

# combine all genre bias plots
grid.arrange(genre1_bias, genre2_bias, genre3_bias, genres_bias)

# movie_year
edx_train %>% group_by(movie_year) %>%
  summarise(b_my = mean(rating - mu)) %>%
  ggplot(aes(b_my)) +
  geom_histogram(bins=15, col="black")

# clear workspace
rm(genre1_bias, genre2_bias, genre3_bias, genres_bias)

####################### Step 4: Predictor Variability  #######################
## Check to see if the data gives too much weight to movies/users with few ratings

## movieId/title
# highest rated movies have < 5 total ratings
unique_movies %>% arrange(desc(avg_rating)) %>%
  select(n, avg_rating)
# lowest rated movies has < 200 total ratings
unique_movies %>% arrange(avg_rating) %>%
  select(n, avg_rating)

# higher uncertainty in these results since they are calculated w/ a very low number of ratings
# see range of n below
range(unique_movies$n)

## userId
# highest ratings have < 40 total users
unique_users %>% arrange(desc(avg_rating)) %>%
  select(n, avg_rating)
# lowest ratings have < 150 total users
unique_users %>% arrange(avg_rating) %>%
  select(n, avg_rating)

# higher uncertainty in these results since they are calculated w/ a very low number of users
# see range of n below
range(unique_users$n)

## genres
# same observations as w/ userId/movieId

unique_genres %>% arrange(desc(avg_rating)) %>%
  select(n_ratings, avg_rating)
unique_genres %>% arrange(avg_rating) %>%
  select(n_ratings, avg_rating)

range(unique_genres$n_ratings)

## dates/timestamp
# same observations as w/ userId/movieId
unique_dates %>% arrange(desc(avg_rating)) %>%
  select(n, avg_rating, date)
unique_dates %>% arrange(avg_rating) %>%
  select(n, avg_rating, date)

range(unique_dates$n)

## In order to reduce effect of variability, will need to use penalized least squares (regularization)
#   for userID, movieID, timestamp/date, and genre_1

####################### Step 5: Prediction Model(s)  #######################

## Separate genres of test set into first 3 genres
u <- edx_test %>% separate(genres, sep = "\\|", into = genre_columns, extra = "drop", fill = "right")
u <- u %>% replace_na(list(genre_1 = no_genre, genre_2 = no_genre, genre_3 = no_genre))

# average rating, mu
mu <- mean(edx_train$rating)

# RMSE equation
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Naive Model (no predictors used)
naive_model_RMSE <- RMSE(edx_test$rating, mu)
naive_model_RMSE

## Un-regularized Models
# NOTE: Should use lm(y ~ x), but since there is so much data, better to use an approximation

### Single effects
# userId (b_u) only
b_u <- edx_train %>% group_by(userId) %>% 
  summarise(b_u = sum(rating - mu)/n())
user_pred <- mu + edx_test %>% left_join(b_u, by = "userId") %>% pull(b_u)
user_rmse <- RMSE(edx_test$rating, user_pred)

# movieId (b_i) only
b_i <- edx_train %>% group_by(movieId) %>% 
  summarise(b_i = sum(rating - mu)/n())
movie_pred <- mu + edx_test %>% left_join(b_i, by = "movieId") %>% pull(b_i)
movie_rmse <- RMSE(edx_test$rating, movie_pred)

# genre (g_ui)
g_ui <- edx_train %>% group_by(genres) %>%
  summarise(g_ui = sum(rating - mu)/n())
genre_pred <- mu + edx_test %>% left_join(g_ui, by = "genres") %>% pull(g_ui)
genre_rmse <- RMSE(edx_test$rating, genre_pred)

# Individual Genres
# genre_1
genre_1_pred <- mu + u %>% left_join(g_ui1, by = "genre_1") %>% pull(g_ui)
genre1_rmse <- RMSE(u$rating, genre_1_pred)

# genre_2
genre_2_pred <- mu + u %>% left_join(g_ui2, by = "genre_2") %>% pull(g_ui)
genre2_rmse <- RMSE(u$rating, genre_2_pred)

# genre_3
genre_3_pred <- mu + u %>% left_join(g_ui3, by = "genre_3") %>% pull(g_ui)
genre3_rmse <- RMSE(u$rating, genre_3_pred)

# timestamp/date (d_ui) only
date_pred <- mu + edx_test %>% mutate(date = as_datetime(timestamp)) %>%
  mutate(date = round_date(date, unit = "week")) %>%
  left_join(unique_dates, by = "date") %>% pull(d_ui)
date_rmse <- RMSE(edx_test$rating, date_pred) # 1.0562

# movie_year only
b_my <- edx_train %>% group_by(movie_year) %>%
  summarise(b_my = mean(rating - mu))

movie_year_pred <- mu + edx_test %>% mutate(movie_year = movie_to_year(title)) %>%
  left_join(b_my, by="movie_year") %>% pull(b_my)
movie_year_rmse <- RMSE(edx_test$rating, movie_year_pred)

# tabulate results
data.frame(predictor = c("none (naive)","userId", "movieId", "genres", "genre_1", "genre_2", "genre_3", "timestamp", "movie_year"),
           model_equation = c("Y = mu", "Y = mu + x_u", "Y = mu + x_i", "Y = mu + x_g",
                              "Y = mu + x_g1", "Y = mu + x_g2", "Y = mu + x_g3", "Y = mu + x_t", "Y = mu + x_my"),
           RMSE = c(naive_model_RMSE, user_rmse, movie_rmse, genre_rmse, genre1_rmse, genre2_rmse, genre3_rmse, date_rmse, movie_year_rmse))

# clear workspace
rm(user_pred, user_rmse, movie_pred, movie_rmse, genre_pred, genre_rmse, genre_1_pred, genre1_rmse,
   genre_2_pred, genre2_rmse, genre_3_pred, genre3_rmse, movie_year_pred, movie_year_rmse,
   date_pred, date_rmse)

### Multiple effects
# userId and movieId
b_i_u <- edx_train %>% group_by(movieId) %>% 
  left_join(b_u, by = "userId") %>%
  summarise(b_i = sum(rating - mu - b_u)/n())
user_movie_pred <- mu + edx_test %>% left_join(b_u, by = "userId") %>% 
  left_join(b_i_u, by = "movieId") %>% 
  mutate(pred = b_u + b_i) %>% pull(pred)
user_movie_rmse <- RMSE(edx_test$rating, user_movie_pred) # 0.88263

# Since there are max 20 unique genres per column, the lm() function can be used
genre_123_pred <- lm(rating ~ as.factor(genre_1) + as.factor(genre_2) + as.factor(genre_3), data = t)
genre_123_rmse <- RMSE(u$rating, predict(genre_123_pred, u)) # 1.0383

# userId, movieId, genres
g_ui <- edx_train %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  group_by(genres) %>%
  summarise(g_ui = sum(rating - b_u - b_i - mu)/n())
pred <- mu + edx_test %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  left_join(g_ui, by = "genres") %>%
  mutate(y = b_u + b_i + g_ui) %>%
  pull(y)
user_movie_genres_rmse <- RMSE(edx_test$rating, pred) # 0.88263

# userId, movieId, movie_year
b_u_i_my <- edx_train %>% left_join(b_i_u, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(movie_year) %>%
  summarise(b_my = sum(rating - b_u - b_i - mu)/n())
pred <- mu + edx_test %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  left_join(b_u_i_my, by = "movie_year") %>%
  mutate(y = b_u + b_i + b_my) %>%
  pull(y)
user_movie_movieyear_RMSE <- RMSE(edx_test$rating, pred) # 0.88263

# userId, movieId, genre_1, genre_2, genre_3
g_ui_1 <- t %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  group_by(genre_1) %>%
  summarise(g_ui1 = sum(rating - mu - b_u - b_i)/n())
g_ui_2 <- t %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  left_join(g_ui_1, by = "genre_1") %>%
  group_by(genre_2) %>%
  summarise(g_ui2 = sum(rating - mu - b_u - b_i - g_ui1)/n())
g_ui_3 <- t %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  left_join(g_ui_1, by = "genre_1") %>%
  left_join(g_ui_2, by = "genre_2") %>%
  group_by(genre_3) %>%
  summarise(g_ui3 = sum(rating - mu - b_u - b_i - g_ui1 - g_ui2)/n())

# effects of g_ui_2 and g_ui_3 are to the 10^-19 power (basically 0)

pred <- mu + u %>% left_join(b_u, by = "userId") %>%
  left_join(b_i_u, by = "movieId") %>%
  left_join(g_ui_1, by = "genre_1") %>%
  left_join(g_ui_2, by = "genre_2") %>%
  left_join(g_ui_3, by = "genre_3") %>%
  mutate(y = b_u + b_i + g_ui1 + g_ui2 + g_ui3) %>%
  pull(y)
user_movie_genre123_rmse <- RMSE(u$rating, pred) # 0.88262

# tabulate results
data.frame(predictors = c("userId, movieId", "genre_1, genre_2, genre_3", "userId, movieId, genres", "userId, movieId, movie_year", "userId, movieId, genre_1, genre_2, genre_3"),
           model_equation = c("Y = mu + x_u + x_i", "Y = mu + x_g1 + x_g2 + x_g3", "Y = mu + x_i + x_u + x_g", "Y = mu + x_i + x_u + x_my", "Y = mu + x_i + x_u + x_g1 + x_g2 + x_g3"),
           RMSE = c(user_movie_rmse, genre_123_rmse, user_movie_genres_rmse, user_movie_movieyear_RMSE, user_movie_genre123_rmse))

# clear workspace
rm(user_movie_rmse, genre_123_rmse, user_movie_genres_rmse, user_movie_movieyear_RMSE, user_movie_genre123_rmse,
   pred, user_movie_pred, genre_123_pred)


## Regularized Models
lambdas <- seq(0, 10, 0.5)

# Single Effect Models
## User effect only
rmses_user <- sapply(lambdas, function(l){
  b_u <- edx_train %>% group_by(userId) %>%
    summarise(b_u = sum(rating - mu)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_u, by = "userId") %>%
    mutate(y_ui = mu + b_u) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

temp <- data.frame(lambdas = lambdas, rmses = rmses_user)
temp %>% ggplot(aes(x=lambdas, y=rmses)) +
  geom_point() +
  xlab("Lambdas") +
  ylab("RMSE") +
  ggtitle("userId Regularized Model, Lambda Tuning") +
  theme(text = element_text(family = "Times New Roman", size=10))

reg_user_rmse <- min(rmses_user) # 0.9778
lambda_user_rmse <- lambdas[which.min(rmses_user)] # lambda = 5.5

## Movie effect only
rmses_movie <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    mutate(y_ui = mu + b_i) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

qplot(lambdas, rmses_movie)
reg_movie_rmse <- min(rmses_movie) # 0.94367
lambda_movie_rmse <- lambdas[which.min(rmses_movie)] # lambda = 2.5

## Movie year only
rmses_movie_year <- sapply(lambdas, function(l){
  b_my <- edx_train %>% group_by(movie_year) %>%
    summarise(b_my = sum(rating - mu)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_my, by = "movie_year") %>%
    mutate(y_ui = mu + b_my) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

qplot(lambdas, rmses_movie)
reg_movieyear_rmse <- min(rmses_movie) # 0.94367
lambda_movieyear_rmse <- lambdas[which.min(rmses_movie)] # lambda = 2.5

# genres
rmses_genres <- sapply(lambdas, function(l){
  # genre (g_ui)
  g_ui <- edx_train %>% group_by(genres) %>%
    summarise(g_ui = sum(rating - mu)/(n()+l))
  genre_pred <- mu + edx_test %>% left_join(g_ui, by = "genres") %>% pull(g_ui)
  
  RMSE(edx_test$rating, genre_pred)
})

qplot(lambdas, rmses_genres)
reg_genre_rmse <- min(rmses_genres) # 1.018
lambda_genres_rmse <- lambdas[which.min(rmses_genres)] # 1

# tabulate results
data.frame(predictor = c("none (naive)","userId", "movieId", "movie_year","genres"),
           model_equation = c("Y = mu", "Y = mu + x_u", "Y = mu + x_i", "Y = mu + x_my", "Y = mu + x_g"),
           lambda = c(0, lambda_user_rmse, lambda_movie_rmse, lambda_movieyear_rmse, lambda_genres_rmse),
           RMSE = c(naive_model_RMSE, reg_user_rmse, reg_movie_rmse, reg_movieyear_rmse, reg_genre_rmse))

# clear workspace
rm(lambda_user_rmse, lambda_movie_rmse, lambda_movieyear_rmse, lambda_genres_rmse,
   naive_model_RMSE, reg_user_rmse, reg_movie_rmse, reg_movieyear_rmse, reg_genre_rmse)

# Multi-Effect Models
## User and Movie Effect
rmses_user_movie <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(y_ui = mu + b_i + b_u) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

qplot(lambdas, rmses_user_movie)
reg_user_movie_rmse <- min(rmses_user_movie) # 0.86524
lambda_user_movie_rmse <- lambdas[which.min(rmses_user_movie)] # lambda = 5

## User, Movie, Genres
rmses_user_movie_genres <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n() + l))
  
  g_ui <- edx_train %>% left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarise(g_ui = sum(rating - mu - b_i - b_u)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(g_ui, by = "genres") %>%
    mutate(y_ui = mu + b_i + b_u + g_ui) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

qplot(lambdas, rmses_user_movie_genres)
reg_user_movie_genre_rmse <- min(rmses_user_movie_genres) # 0.86494
lambda_movie_genre_rmse <- lambdas[which.min(rmses_user_movie_genres)] # lambda = 5

## User, Movie, Movie Year
rmses_user_movie_year <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n() + l))
  
  b_my <- edx_train %>% left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(movie_year) %>%
    summarise(b_my = sum(rating - mu - b_i - b_u)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_my, by = "movie_year") %>%
    mutate(y_ui = mu + b_i + b_u + b_my) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

qplot(lambdas, rmses_user_movie_year)
reg_user_movie_year_rmse <- min(rmses_user_movie_year) # 0.86498
lambda_user_movie_year_rmse <- lambdas[which.min(rmses_user_movie_year)] # lambda = 4.5

## User, Movie, Genre_1, Genre_2, Genre_3
rmses_user_movie_genre123 <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n() + l))
  
  g_ui1 <- t %>% left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genre_1) %>%
    summarise(g_ui1 = sum(rating - mu - b_i - b_u)/(n() + l))
  
  g_ui2 <- t %>% left_join(b_u, by = "userId") %>%
    left_join(b_i_u, by = "movieId") %>%
    left_join(g_ui_1, by = "genre_1") %>%
    group_by(genre_2) %>%
    summarise(g_ui2 = sum(rating - mu - b_u - b_i - g_ui1)/n())
  
  g_ui3 <- t %>% left_join(b_u, by = "userId") %>%
    left_join(b_i_u, by = "movieId") %>%
    left_join(g_ui_1, by = "genre_1") %>%
    left_join(g_ui_2, by = "genre_2") %>%
    group_by(genre_3) %>%
    summarise(g_ui3 = sum(rating - mu - b_u - b_i - g_ui1 - g_ui2)/n())
  
  pred <- u %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(g_ui1, by = "genre_1") %>%
    left_join(g_ui2, by = "genre_2") %>%
    left_join(g_ui3, by = "genre_3") %>%
    mutate(y_ui = mu + b_i + b_u + g_ui1 + g_ui2 + g_ui3) %>%
    pull(y_ui)
  
  return(RMSE(u$rating, pred))
})

qplot(lambdas, rmses_user_movie_genre123)
reg_user_movie_genre123 <- min(rmses_user_movie_genre123) # 0.86563
lambda_user_movie_genre123 <- lambdas[which.min(rmses_user_movie_genre123)] # lambda = 4.5

## User, Movie, Genre_1
rmses_user_movie_genre1 <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n() + l))
  
  g_ui1 <- t %>% left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genre_1) %>%
    summarise(g_ui1 = sum(rating - mu - b_i - b_u)/(n() + l))
  
  pred <- u %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(g_ui1, by = "genre_1") %>%
    mutate(y_ui = mu + b_i + b_u + g_ui1) %>%
    pull(y_ui)
  
  return(RMSE(u$rating, pred))
})

qplot(lambdas, rmses_user_movie_genre1)
reg_user_movie_genre1_rmse <- min(rmses_user_movie_genre1) # 0.86514
lambda_user_movie_genre1_rmse <- lambdas[which.min(rmses_user_movie_genre1)] # lambda = 5

## User, Movie, Movie Year, Genres
rmses_user_movie_year_genres <- sapply(lambdas, function(l){
  b_i <- edx_train %>% group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() + l))
  
  b_u <- edx_train %>% left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i)/(n() + l))
  
  b_my <- edx_train %>% left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(movie_year) %>%
    summarise(b_my = sum(rating - mu - b_i - b_u)/(n() + l))
  
  g_ui <- edx_train %>% left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_my, by = "movie_year") %>%
    group_by(genres) %>%
    summarise(g_ui = sum(rating - mu - b_i - b_u - b_my)/(n() + l))
  
  pred <- edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_my, by = "movie_year") %>%
    left_join(g_ui, by = "genres") %>%
    mutate(y_ui = mu + b_i + b_u + b_my + g_ui) %>%
    pull(y_ui)
  
  return(RMSE(edx_test$rating, pred))
})

qplot(lambdas, rmses_user_movie_year_genres)
reg_user_movie_year_genres_rmse <- min(rmses_user_movie_year_genres) # 0.86475
lambda_user_movie_year_genres_rmse <- lambdas[which.min(rmses_user_movie_year)] # lambda = 4.5

# tabulate results
data.frame(predictors = c("none (naive)","userId, movieId", "userId, movieId, genres", "userId, movieId, movie_year", 
                          "userId, movieId, genre_1, genre_2, genre_3", "userId, movieId, genre_1", 
                          "userId, movieId, genres, movie_year"),
           model_equation = c("Y = mu", "Y = mu + x_u + x_i", "Y = mu + x_u + x_i + x_g", "Y = mu + x_u + x_i + x_my", 
                              "Y = mu + x_u + x_i + x_g1 + x_g2 + x_g3",
                              "Y = mu + x_u + x_i + x_g1", "Y = mu + x_u + x_i + x_g + x_my"),
           lambda = c(0, lambda_user_movie_rmse, lambda_movie_genre_rmse, lambda_user_movie_year_rmse, 
                      lambda_user_movie_genre123, lambda_user_movie_genre1_rmse, lambda_user_movie_year_genres_rmse),
           RMSE = c(naive_model_RMSE, reg_user_movie_rmse, reg_user_movie_genre_rmse, reg_user_movie_year_rmse, 
                    reg_user_movie_genre123, reg_user_movie_genre1_rmse, reg_user_movie_year_genres_rmse))

####################### Step 6: RMSE & Validation #######################

## Selected Model
# Predictors: User, Movie, Genres, Movie Year
lambda <- 4.5
mu <- mean(edx_train$rating)

# calculate the effects of each predictor
movie_effect <- edx_train %>% group_by(movieId) %>% 
  summarise(b_i = sum(rating - mu)/(n() + lambda))
user_effect <- edx_train %>% left_join(movie_effect, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i)/(n() + lambda))
movie_year_effect <- edx_train %>% left_join(user_effect, by = "userId") %>%
  left_join(movie_effect, by = "movieId") %>%
  group_by(movie_year) %>%
  summarise(b_my = sum(rating - mu - b_u - b_i)/(n() + lambda))
genres_effect <- edx_train %>% left_join(user_effect, by = "userId") %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(movie_year_effect, by = "movie_year") %>%
  group_by(genres) %>%
  summarise(g_ui = sum(rating - mu - b_u - b_i - b_my)/(n() + lambda))

# predict ratings on the validation set
model <- validation %>% left_join(user_effect, by = "userId") %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(genres_effect, by = "genres") %>%
  left_join(movie_year_effect, by = "movie_year") %>%
  mutate(pred = mu + b_i + b_u + g_ui + b_my) %>%
  pull(pred)

## Validation RMSE
RMSE(validation$rating, model) # 0.86516
