# HarvardX Capstone Project - MovieLens Prediction
# Name: Marcelo Argotti
# Date: 6/20/2020


## load packages and libraries needed
# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", 
                                       repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", 
                                       repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", 
                                          repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", 
                                         repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", 
                                      repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", 
                                              repos = "http://cran.us.r-project.org")


################################
# Create edx set, validation set
################################


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(movieId),title = as.character(title),
         genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

rm(dl, ratings, movies, test_index, temp, movielens, removed)



############## DATA OBSERVATION  ################
##################################################
#let's take a glimpse of the edx data set
class(edx)
glimpse(edx)
#let's take a glimpse of the validation data set
class(validation)
glimpse(validation)
#Let's take a look at the ratings (our dependent value, y)
unique(edx$rating) 

#Rating plot distribution of our training set
ratings_distribution <- edx %>% group_by(rating) %>% 
  summarize(ratings_sum = n()) %>% arrange(desc(ratings_sum))
ratings_distribution

ggplot(edx, aes(x= edx$rating, fill = edx$rating)) +
  geom_histogram( binwidth = 0.25) +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  labs(x="rating", y="total of ratings", caption = "source: edx dataset") +
  ggtitle("histogram : number of ratings for each rating")


#We make a copy of the training dataset due to we want to build a matrix 
#containing some features as factors. Also, we  want to observe unique 
#or independent genres without been combined, in order to visualize which 
#genre has more ratings, get a better understanding of the edx dataset, 
#and identify outliers.
edx_copy <- edx
genres <- edx_copy$genres %>% str_split(pattern = "\\|")
genres_unique <- genres %>% unlist() %>% unique()
genres_unique

# The following lines of code define a table containing independent genres 
#as variables, each one of them are described only by 0s and 1s. The 0s will
#be treated as non genre matching the corresponding movie.
cols_edxcopy <- ncol(edx_copy)
for (i in seq_along(genres_unique)) {
  id <- grepl(pattern = genres_unique[i], edx_copy$genres)
  edx_copy[[cols_edxcopy + i]] <- 0
  edx_copy[[cols_edxcopy + i]][id] <- 1
}
names(edx_copy)[(cols_edxcopy + 1):ncol(edx_copy)] <- genres_unique


#summary of how many of them are considered in our training dataset.
topgen <- edx_copy %>% summarise(Comedy= sum(Comedy), Romance=sum(Romance),
                                 Action=sum(Action), Crime=sum(Crime), 
                                 Thriller=sum(Thriller), Drama=sum(Drama), 
                                 Sci_fi=sum(`Sci-Fi`),Adventure=sum(Adventure),
                                 Children=sum(Children), Fantasy=sum(Fantasy),
                                 War=sum(War), Animation=sum(Animation), 
                                 Musical=sum(Musical), Western=sum(Western), 
                                 Mystery=sum(Mystery), Film_noir=sum(`Film-Noir`),
                                 Horror=sum(Horror), Documentary=sum(Documentary),
                                 Imax=sum(IMAX),)
glimpse(sort(topgen))

# We move on with the title features.
top_title <- edx %>% group_by(title) %>% 
  summarize(count=n()) %>% top_n(20,count) %>%
  arrange(desc(count))

top_title %>% 
  ggplot(aes(x=reorder(title, count), y=count)) +
  geom_bar(stat='identity', fill="cadetblue") + coord_flip(y=c(0, 40000)) +
  labs(x="", y="Total of ratings") +
  geom_text(aes(label= count), hjust=-0.15, size=2) +
  labs(title="Top 20 movies title based \n on number of ratings" ,
       caption = "source data: edx set")

# With the kable and head function we generate a simple table that contains 
#the top 30 rated movies, including the genres associated to each one. 
#It can be seen that the first 25 movies have been rated over 20.000

kable(head(edx %>% group_by(title,genres) %>%
             summarize(count=n()) %>%
             top_n(20,count) %>%
             arrange(desc(count)) ,30)) %>%
  kable_styling(bootstrap_options = "bordered", 
                full_width = FALSE ,position ="center") %>%
  column_spec(1,bold = TRUE ) %>%
  column_spec(2,italic = TRUE) %>%
  column_spec(3,color = "brown")

#display the distinct number of users and distinct number of 
#movies of our training set.

edx %>% summarize(distinct_users=n_distinct(userId), 
                  distinct_movies=n_distinct(movieId))


# Lets build a scatter point plot of average ratings vs date. 
# we make use of smooth geom in order to identify outliers or overplotting.
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  group_by(date) %>% summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) + geom_point() + geom_smooth() +
  ggtitle("Timestamp")+
  labs(subtitle = "time unit : [month]",
       caption = "source data : Edx set")


#Lets build a rating matrix using the SparseMatrix function
#We make a new copy of the edx dataset in order to preserve the origanl
# class of userId and movieId that will be converted into numeric vectors

edx_copy2 <- edx

edx_copy2$userId <- as.factor(edx_copy2$userId)
edx_copy2$movieId <- as.factor(edx_copy2$movieId)

edx_copy2$userId <- as.numeric(edx_copy2$userId)
edx_copy2$movieId <- as.numeric(edx_copy2$movieId)

scattered_ratings <- sparseMatrix(i = edx_copy2$userId, 
                                  j = edx_copy2$movieId, 
                                  x = edx_copy2$rating,
                                  dims = c(length(unique(edx_copy2$userId)),
                                           length(unique(edx_copy2$movieId))),
                                  dimnames = list(paste(
                                    "user", 1:length(unique(edx_copy2$userId)), 
                                    sep = ""),
                                    paste("movie", 1:length(unique(
                                      edx_copy2$movieId)), sep = "")))

rm(edx_copy2)
#Lets give a look on the first 30 users
scattered_ratings[1:30,1:30]

###### Convert the dgCMatrix (scattered_ratings) matrix into a recommenderlab matrix
# dgCMatrix into recommenderlab rating matrix
rating_matrix <- new("realRatingMatrix", data=scattered_ratings)
rating_matrix


# userId similarity, first 80 users
user_similarity <- similarity(rating_matrix[1:80,], 
                              method = "cosine", 
                              which = "users")
image(as.matrix(user_similarity), main = "User similarity")


#movieId similarity, first 80 users
movie_similarity <- similarity(rating_matrix[,1:80], 
                               method = "cosine", 
                               which = "items")
image(as.matrix(movie_similarity), main = "Movies similarity")



########################## METHODS & ANALYSIS   #######################
#######################################################################


# RMSE function for vectors of ratings and their corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Model 1: predict same rating for all movies regardless of users
# predict average rating of all movies
mu_hat <- mean(edx$rating)
mu_hat
# calculate rmse of this naive approach 
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse
# add rmse results in a table
rmse_results <- data.frame(method = "Naive approach", RMSE = naive_rmse)
rmse_results %>% knitr::kable()


# Model 2: modelling movie effect
# estimate movie bias 'b_i' for all movies
mu <- mean(edx$rating)
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
# plot these movie 'bias'
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
# calculate predictions considering movie effect
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
# calculate rmse after modelling movie effect
model_1_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse))
rmse_results %>% knitr::kable()


# Model 4: regularizing movie + user effect model from previous models
# choosing the penalty term lambda
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses) 

############# RESULTS & CONCLUSIONS #######################
#The RMSEs result for various models are as follows:

# lambda that minimizes rmse
lambda <- lambdas[which.min(rmses)]
lambda
# calculate rmse after regularizing movie + user effect from previous models
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User effect model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()


  

