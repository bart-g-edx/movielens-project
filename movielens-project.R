#
# title: "MovieLens Project"
# author: "bart-g"
# date: "4/8/2020"
#
library(tidyverse)
library(caret)
library(data.table)
library(timeDate)
library(knitr)

##################################
# Prerequisite data preparation
# - as originally provided on course website

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1) # , sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation, by = c("userId", "movieId", "rating", "timestamp", "title", "genres"))
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################
# 1. Overview 

# snapshot of the records available

head(edx) %>% knitr::kable()

# summary of users and movies sample populations 

tibble(unique_users = n_distinct(edx$userId), unique_movies = n_distinct(edx$movieId)) %>% 
    knitr::kable(col.names = c("Unique users", "Movies being rated"))

###############################
# 2. Analysis

# error function, the RMSE

RMSE <- function(actual_outcomes, predicted_outcomes){
    sqrt(mean((actual_outcomes - predicted_outcomes)^2))
}

# partitioning of the edx set

set.seed(1) 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# required data re-joining,
# so final test set is representative of train set movies and users

test_set <- test_set %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
rm(test_index)

################################
# 1st model: intercept

# plot distribution

train_set %>% ggplot(aes(rating)) +
    geom_histogram(aes(y = ..density..), binwidth = 1, color = "grey", fill = "lightblue") +
    geom_density(alpha = 0.1, bw=0.5, fill = "red") + 
    ggtitle("Ratings histogram with density")

# moments of distribution

tibble(skewness(train_set$rating) , kurtosis(train_set$rating)) %>% 
    knitr::kable(col.names = c("skewnes", "kurtosis"))

# plot ratings, note the mass of distribution and mean being positive

train_set %>% ggplot(aes(rating)) +
    geom_histogram(binwidth = 0.5, color = "grey", fill = "lightblue") +
    ggtitle("Ratings histogram at half-points")

# calculate intercept only model

mu_hat <- mean(train_set$rating)
rmse1 <- RMSE(test_set$rating, mu_hat)

mu_hat  %>% knitr::kable(col.names = "Intercept")

rmse1  %>% knitr::kable(col.names = "Model RMSE")

################################
# 2nd model: movie effect

# calculate movie effect

mu_hat <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- mu_hat + test_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    .$b_i

rmse2 <- RMSE(predicted_ratings, test_set$rating)

# plot movie effect

movie_avgs %>% ggplot(aes(b_i)) +
    geom_histogram(binwidth = 0.5, color = "grey", fill = "lightblue") +
    ggtitle("Movie effect histogram") +
    xlab("rating change") +
    ylab("count")

rmse2  %>% knitr::kable(col.names = "Model RMSE")

#########################################
# 3rd model: movie + user effect

# calculate user effect

user_avgs <- train_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- test_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    .$pred

rmse3 <- RMSE(predicted_ratings, test_set$rating)

# plot user effect

user_avgs %>% ggplot(aes(b_u)) +
    geom_histogram(binwidth = 0.5, color = "grey", fill = "lightblue") +
    ggtitle("User effect histogram") +
    xlab("rating change") +
    ylab("count")

rmse3  %>% knitr::kable(col.names = "Model RMSE")

###################################################
# 4th model: user + movie + genres effect

# different genres comparison

res <- sapply(c("Action", "Drama", "Comedy", "Crime", "Sci-Fi", "Thriller", "Romance"), function(selected_genres) {
    edx %>% filter(grepl(selected_genres, genres)) %>%
        summarize(total = n(), avg = mean(rating))
})
res_columns <- dim(res)[2]
res[2,1:res_columns] <- round(as.numeric(res[2,1:res_columns]),6)
res %>% knitr::kable()

# calculate genres effect

genres_avgs <- train_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu_hat - b_i - b_u))

predicted_ratings <- test_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genres_avgs, by='genres') %>%
    mutate(pred = mu_hat + b_i + b_u + b_g) %>%
    .$pred

rmse4 <- RMSE(predicted_ratings, test_set$rating)

# plot genres effect

genres_avgs %>% ggplot(aes(b_g)) +
    geom_histogram(binwidth = 0.25, color = "grey", fill = "lightblue") +
    ggtitle("Genres effect histogram") +
    xlab("rating change") +
    ylab("count")

rmse4  %>% knitr::kable(col.names = "Model RMSE")

#################################################
# Regularisation of the best model

# run regularisation with lambda sequence

lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
rmses <- sapply(lambdas, function(lambda_arg){
    b_i <- train_set %>%
        group_by(movieId) %>%
        summarize(b_i = sum(rating - mu)/(n()+lambda_arg))
    b_u <- train_set %>% 
        left_join(b_i, by="movieId") %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - mu - b_i)/(n()+lambda_arg))
    b_g <- train_set %>% 
        left_join(b_i, by='movieId') %>%
        left_join(b_u, by='userId') %>%
        group_by(genres) %>%
        summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda_arg))
    # complete model
    predicted_ratings <- 
        test_set %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_g, by = "genres") %>%
        mutate(estimate = mu + b_i + b_u + b_g) %>%
        .$estimate
    last_rmse <- RMSE(predicted_ratings, test_set$rating)
    return(last_rmse)
})

rmse_lambdas <- tibble(rmse = rmses, lambda = lambdas)
rmse_lambdas %>% ggplot(aes(x = lambdas, y = rmses)) +
    geom_line(color = "red") +
    geom_point(color = "blue") +
    ggtitle("Minimization of residuals") +
    xlab("Lambda") +
    ylab("RMSE")

lambda <- lambdas[which.min(rmses)]
lambda  %>% knitr::kable(col.names = "Regularisation Lambda")

rmseR <- rmses[which.min(rmses)]
rmseR  %>% knitr::kable(col.names = "RMSE")

##############################
# 3. Results

lambda %>% knitr::kable(col.names = "Lambda")

# validation set RMSE calculation

mu <- mean(edx$rating)
b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))
b_g <- edx %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda))
predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
validation_rmse <- RMSE(predicted_ratings, validation$rating)
validation_rmse  %>% knitr::kable(col.names = "RMSE")

# calculate raw effects with lambda = 0, unregularised

b_i_raw <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()), n_count = n())
b_u_raw <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()), n_count = n())
b_g_raw <- edx %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()), n_count = n())

# combine regularised effect with raw effects for qq-plotting

reg_u_data <- tibble(regularised = b_u_raw$b_u, raw = b_u$b_u, n = b_u_raw$n_count)
reg_i_data <- tibble(regularised = b_i_raw$b_i, raw = b_i$b_i, n = b_i_raw$n_count)
reg_g_data <- tibble(regularised = b_g_raw$b_g, raw = b_g$b_g, n = b_g_raw$n_count)
rm(b_i, b_u, b_g, b_i_raw, b_u_raw, b_g_raw)

reg_u_data %>% ggplot(aes(raw, regularised, size = sqrt(n))) + 
    geom_point(shape = 1, alpha = 0.5, color = "blue") +
    ggtitle("Regularised user effect vs unregularised") + 
    xlab("unregularised")

reg_i_data %>% ggplot(aes(raw, regularised, size = sqrt(n))) + 
    geom_point(shape = 1, alpha = 0.5, color = "blue") +
    ggtitle("Regularised movie effect vs unregularised") + 
    xlab("unregularised")

reg_g_data %>% ggplot(aes(raw, regularised, size = sqrt(n))) + 
    geom_point(shape = 1, alpha = 0.5, color = "blue") +
    ggtitle("Regularised genres effect vs unregularised") + 
    xlab("unregularised")
rm(mu, reg_u_data, reg_i_data, reg_g_data)

###########################
# Models comparison

# models comparison, indicated improvements in rating error

all_results <- data.frame(
    model_type = c("intercept only", 
                   "movie effect", 
                   "movie + user effect", 
                   "movie + user + genres effect", 
                   "regularized 'movie + user + genres effect'", 
                   "regularized 'movie + user + genres effect' (validation set)"), 
    rmse = c(rmse1, rmse2, rmse3, rmse4, rmseR, validation_rmse))
all_results <- all_results %>% mutate(imp = (lag(rmse)-rmse))
all_results$imp[1] <- all_results$rmse[1]
all_results %>% knitr::kable(col.names = c("Model Type", "RMSE", "Improvement"))

validation_rmse
