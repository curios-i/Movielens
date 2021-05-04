##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

#tcrossprod() is used from library Matrix, while funkSVD() is used from library recommenderlab

library(tidyverse)
library(caret)
library(data.table)
library(Matrix)
library(recommenderlab)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),title = as.character(title),            genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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
###########################################################
memory.limit(size=200000) #sets virtual memory size to 200,000Mb to process large data
set.seed(1, sample.kind="Rounding")
#Partition edx into 80% edx_train and 20% edx_test so that we can tune the model without touching the validation data
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train<-edx[-test_index,]
temp<-edx[test_index,]

#make sure that the movieId and userId in edx_train are also in edx_test
edx_test<-temp%>%semi_join(edx_train,by="movieId")%>%semi_join(edx_train,by="userId")
#Add removed records from edx_test back to edx_train
removed<-anti_join(temp,edx_test)
edx_train<-rbind(edx_train,removed)
#Remove temporary variables to free up the memorylib
rm(removed,temp,test_index)
#################################################################
# defines RMSE function to calculate root mean square error
#################################################################
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
##################################################
#SOLUTION#1
##################################################
# The model for predicted ratings have four components, calculated from edx_train
#1. mu - mean of ratings
#2 b_i - average rating of a movie i, with L2 regularization
#3 b_u - average rating given by a user I, with L2 regularization
#4 pred - residual ratings calculated from above and then from matrix factorization
mu<-mean(edx_train$rating)
lambdas <- seq(3, 7, 0.25)
# calculates rmses for the above values of lambdas to tune the model
rmses<-sapply(lambdas,function(l){
  b_i<-edx_train%>%group_by(movieId)%>%summarise(b_i=sum(rating-mu)/(n()+l))
  b_u<-edx_train%>%left_join(b_i,by="movieId")%>%
    group_by(userId)%>%
    summarise(b_u=sum(rating-b_i-mu)/(n()+l))
  predicted_ratings_test<-edx_test%>%left_join(b_i,by="movieId")%>%
    left_join(b_u,by="userId")%>%
    mutate(pred=mu+b_i+b_u)%>%
    pull(pred)
  return(RMSE(predicted_ratings_test,edx_test$rating))})
min(rmses)
#[1] 0.8652421
#assigns lambda the value associated with min rmse
lambda<-lambdas[which.min(rmses)]
# 4.75
# calculates b_i and b_u for edx_train. Since all users and movies in edx_test and validation are present in edx_train, these values of b_i and b_u are used later  to calculate the predicted ratings for those datasets
b_i<-edx_train%>%group_by(movieId)%>%summarise(b_i=sum(rating-mu)/(n()+lambda))
b_u<-edx_train%>%left_join(b_i,by="movieId")%>%group_by(userId)%>%
  summarise(b_u=sum(rating-b_i-mu)/(n()+lambda))
# calculates predicted ratings for the edx_test based on b_i and b_u
predicted_ratings_test<-edx_test%>%left_join(b_i,by="movieId")%>%
  left_join(b_u,by="userId")%>%
  mutate(pred=mu+b_i+b_u)%>%
  pull(pred)
#calculates predicted ratings for edx_train, so that we can calculate residuals for further matrix factorization
predicted_ratings_train<-edx_train%>%left_join(b_i,by="movieId")%>%
  left_join(b_u,by="userId")%>%
  mutate(pred=mu+b_i+b_u)%>%
  pull(pred)
#calculated residual ratings after subtracting the predicted ratings from edx_train ratings
edx_train<-edx_train%>%mutate(predicted_ratings_train=predicted_ratings_train,resid=rating-predicted_ratings_train)
# contructs a matrix y to run svd
y<-edx_train%>%select(userId,movieId,resid)%>%spread(movieId,resid)%>%as.matrix()
rownames(y)<- y[,1]
y<-y[,-1]
# gammas are L2 regularizatino terms in svd stochastic gradient descent algoritm
# we shall calculate rmses for each of these gammas and will determin which one gives optimal tuning
gammas<-c(0.015,0.025,0.035)
# f_rmses stores rmses calculated for respective values of gammas
f_rmses<-c(1,1,1)
###############################################################################
# runs the Simon Funk's stocastic gradient descent algorithm to factorize matrix of residuals
# this will take several hours
fsvd<-funkSVD(y, k=3, gamma=gammas[1], lambda=0.001, verbose=TRUE)
# y_hat is prediction matrix from SVD 
y_hat_0.015<-tcrossprod(fsvd$U,fsvd$V)
# Assigns row and column names to prediction matrix so that predicted rating can be pulled from the matrix for any user i and movie j
rownames(y_hat_0.015)<-rownames(y)
colnames(y_hat_0.015)<-colnames(y)
# creates a placeholder for pred = predicted ratings for residuals from SVD
pred<-rep(0,length(edx_test$userId))
# fill vector pred from the prediction matrix for respective userId and movieId
for(i in 1:length(edx_test$userId)){
  pred[i]<-y_hat_0.015[as.character(edx_test$userId[i]),as.character(edx_test$movieId[i])]}
# calculated predicted ratings form edx_test for that particular gamma
predicted_ratings_0.015<-predicted_ratings_test+pred
# calculates rmse for that particular gamma
f_rmses[1]<-RMSE(predicted_ratings_0.015,edx_test$rating)
#[1] 0.8260231
###############################################################
# runs the Simon Funk's stocastic gradient descent algorithm to factorize matrix of residuals
fsvd<-funkSVD(y, k=3, gamma=gammas[2], lambda=0.001, verbose=TRUE)
# y_hat is prediction matrix from SVD 
y_hat_0.025<-tcrossprod(fsvd$U,fsvd$V)
# Assigns row and column names to prediction matrix so that predicted rating can be pulled from the matrix for any user i and movie j
rownames(y_hat_0.025)<-rownames(y)
colnames(y_hat_0.025)<-colnames(y)
pred<-rep(0,length(edx_test$userId))
# fill vector pred from the prediction matrix for respective userId and movieId
for(i in 1:length(edx_test$userId)){
  pred[i]<-y_hat_0.025[as.character(edx_test$userId[i]),as.character(edx_test$movieId[i])]}
# calculated predicted ratings form edx_test for that particular gamma
predicted_ratings_0.025<-predicted_ratings_test+pred
# calculates rmse for that particular gamma
f_rmses[2]<- RMSE(predicted_ratings_0.025,edx_test$rating)
#0.8257668
###########################################################################
# runs the Simon Funk's stocastic gradient descent algorithm to factorize matrix of residuals
fsvd<-funkSVD(y, k=3, gamma=gammas[3], lambda=0.001, verbose=TRUE)
y_hat_0.035<-tcrossprod(fsvd$U,fsvd$V)
rownames(y_hat_0.035)<-rownames(y)
colnames(y_hat_0.035)<-colnames(y)
pred<-rep(0,length(edx_test$userId))
for(i in 1:length(edx_test$userId)){
  pred[i]<-y_hat_0.035[as.character(edx_test$userId[i]),as.character(edx_test$movieId[i])]}
predicted_ratings_0.035<-predicted_ratings_test+pred
f_rmses[3]<- RMSE(predicted_ratings_0.035,edx_test$rating)
#[1] 0.8258410
#calculates the optimal gamma from rmses results
gamma_min<-gammas[which.min(f_rmses)]
# assings the final predicttion matrix from the min rmse gamma
y_hat_final<-y_hat_0.025
################################################################################
# In the following section, the predicted ratings from edx_train and tuned from edx_test are now 
# used to calculate the predicted ratings for validation and calculate RMSE
################################################################################
# predicted ratings for validation set is first calculated from mu, b_i and b_u from edx_train
predicted_ratings_valid<-
  validation%>%left_join(b_i,by="movieId")%>%
  left_join(b_u,by="userId")%>%
  mutate(pred=mu+b_i+b_u)%>%
  pull(pred)
# This finds rmse before applying SVD to residual
RMSE(validation$rating,predicted_ratings_valid)
#0.8657012
# A residual vector placeholder for each user in the validation
pred<-rep(0,length(validation$userId))
# pulls the predicted value of residual from the prediction matrix 
for(i in 1:length(validation$userId)){
  pred[i]<-y_hat_final[as.character(validation$userId[i]),as.character(validation$movieId[i])]}
#calculates the final predicted ratings for validation set
predicted_ratings_valid<-predicted_ratings_valid+pred
# calculates final rmse
RMSE(predicted_ratings_valid,validation$rating)
#[1] 0.8252377
#################################################
#SOLUTION#2
#################################################
install.packages("recosystem")
library(recosystem)
r<-Reco()
#In the next statement we tune the model for 5 features or 10 features
#with 3 different values of learning rates (0.05,0.1,0.15)
#for 20 iterations and 5 fold cross validation
opts<-r$tune(data_memory(edx$userId,edx$movieId,rating = edx$rating,index1 = TRUE),opts = list(dim=c(5,10),lrate=c(0.05,0.1,0.15),niter=20,nfold=5,verbose=FALSE))
# opts$min gives us the optimized tuning rate
opts$min
# i.e. 10 features, L1 and L2 regularization factors and a
#learning rate of 0.15
#Following trains on edx data set based on opt$min
r$train(data_memory(edx$userId,edx$movieId,rating = edx$rating,index1 = TRUE),opts = c(opts$min,nthread=1,niter=100))
# Now we use the predict function to get predicted ratings
# for the validation set
predicted_ratings<-r$predict(data_memory(validation$userId,validation$movieId,rating = NULL,index1 = TRUE),out_memory())
#Finally the RMSE function gives the root mean square error
RMSE(predicted_ratings,validation$rating)
#0.7928682