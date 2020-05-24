setwd("C:/Users/DELL/Desktop/Akshay/Group Assignments/Group Assignment 7 ML")
getwd()
cars_data <- read.csv("C:/Users/DELL/Desktop/Akshay/Group Assignments/Group Assignment 7 ML/Cars.csv")
str(cars_data)
library(caret)
library(DMwR)
### Converting variables Engineer, MBA and license in to factor variables
### since they are of categorical type
cars_data$Engineer<-as.factor(cars_data$Engineer)
cars_data$MBA<-as.factor(cars_data$MBA)
cars_data$license<-as.factor(cars_data$license)
cars_data<-knnImputation(cars_data)

### Basic data summary, Univariate, Bivariate analysis, graphs 
barplot(table(cars_data$Transport),main = "Overall Transport",col="Blue",border="Red",
        density=100,ylim = c(0,3000))
summary(cars_data$Transport)
## From summary of churn rate of customers, it can be seen that 
## Number of employees who use car as a mode of transport are 61. 
## i.e 61/444 = 0.137387 (13.7387%)

## Now, considering only factor variables,
barplot(table(cars_data$Engineer),density = 100,col="red",main = "0 - not Engineer, 1 - Engineer",
        ylim = c(0,3500))
summary(cars_data$Engineer)

barplot(table(cars_data$MBA),density = 100,col="orange", main = "0 - Not Done MBA, 1 - DOne MBA",
        ylim = c(0,3500))
summary(cars_data$MBA)

barplot(table(cars_data$license),density = 100,col="purple",main = "0 - Don't Have License, 1 - Have License",
        ylim = c(0,3500))
summary(cars_data$license)

### Check for Multicollinearity - Plot the graph based on Multicollinearity
## For checking multi-collinearity of the dataset, considering only numeric variables.
## Therefore, discarding the factor variable columns
numericdata<-cars_data[,c(-2,-3,-4,-8,-9)]
print(cor(numericdata),digits = 3)
library(corrplot)
corrplot(cor(numericdata), method = c("number"), type = c("full"))

## taking boxplots
boxplot(cars_data$Age ~ cars_data$Engineer,cars_data$MBA, main = "Age vs Engineer", col = 'orange', horizontal = TRUE)
boxplot(cars_data$Age ~ cars_data$MBA, main = "Age vs MBA", col = 'orange', horizontal = TRUE)
## Let us see the avg difference in salary for two profession
boxplot(cars_data$Salary ~cars_data$Engineer, main = "Salary vs Eng.", col = "blue", horizontal = TRUE)
boxplot(cars_data$Salary ~cars_data$MBA, main = "Salary vs MBA.", col = "Green", horizontal = TRUE)
hist(cars_data$Work.Exp, col = "light blue", main = "Distribution of work exp among Employees")
## This is skewed towards right, again this would be on expected lines 
## as there would be more juniors then seniors in any firm
table(cars_data$license,cars_data$Transport)
boxplot(cars_data$Work.Exp ~ cars_data$Gender, main = "Gender Distribution among Employees", col = 'pink', horizontal = TRUE)

## Hypotheses Testiing
boxplot(cars_data$Salary~cars_data$Transport, main="Salary vs Transport", horizontal = TRUE, col = "blue")
boxplot(cars_data$Age~cars_data$Transport, main="Age vs Transport", horizontal = TRUE, col = "blue")
## As was the case with salary, we could see clear demarcation in usage of transport.
## With lower age group 2-wheeler is preferable and with higher work exp car is preferred.
boxplot(cars_data$Distance~cars_data$Transport, main="Distance vs Transport", horizontal = TRUE, col = "blue")
table(cars_data$Gender,cars_data$Transport)

## Our primary interest as per problem statement is to understand the factors influencing car usage. 
## Hence we will create a nwe column for Car usage.
## It will take value 0 for Public Transport & 2 Wheeler and 1 for car usage 
cars_data$CarUsage<-ifelse(cars_data$Transport =='Car',1,0)
table(cars_data$CarUsage)
sum(cars_data$CarUsage == 1)/nrow(cars_data)
cars_data$CarUsage<-as.factor(cars_data$CarUsage)
View(cars_data)
# The number of records for people travelling by car is in minority.
# Hence we need to use an appropriate sampling method on the train data.
# using SMOTE We will use logistic regression, boosting, KNN and NB

##Split the data into test and train
set.seed(123)
cars_split<-createDataPartition(cars_data$CarUsage, p=0.7,list = FALSE,times = 1)
cars_train<-cars_data[cars_split,]
cars_test<-cars_data[-cars_split,]
prop.table(table(cars_train$CarUsage))

## discard transport column from cars_data since we are utilising newly created column CarUsage
cars_train<-cars_train[,c(1:8,10)]
cars_test<-cars_test[,c(1:8,10)]
dim(cars_train)
dim(cars_test)
## The train and test data have almost same percentage of cars usage as the base data
## Apply SMOTE on Training data set
library(DMwR)
cars_SMOTE<-SMOTE(cars_train$CarUsage~., cars_train, perc.over = 250,perc.under = 150)
prop.table(table(cars_SMOTE$CarUsage))
# there is equal split in the data between car users and non car users

## Building Logistic Regression Model
## Create control parameter for GLM
outcome_variable <-'CarUsage'
regressors<-c("Age","Work.Exp","Salary","Distance","license","Engineer","MBA","Gender")
trainctrl<-trainControl(method = 'repeatedcv',number = 10,repeats = 3)
glm<-train(cars_SMOTE[,regressors],cars_SMOTE[,outcome_variable],
               method = "glm", family = "binomial",trControl = trainctrl)
summary(glm$finalModel)
glmcoeff<-exp(coef(glm$finalModel))
glmcoeff
varImp(object = glm)
plot(varImp(object = glm), main="Vairable Importance for Logistic Regression")
# From the model we see that Age and License are more significant.
# When we look at the odds and probabilities table,
# we get to see that Increase in age by 1 year implies that thre is a 98% probability that the employee will use a car. 
# As expected , if the employee has a license, then it implies a 99% probability that he/she will use a car.
# One lkah increase in salary increases the probability of car usage by 72% 
# The null deviance of this model is 357.664 and the residual deviance is 17.959.
# This yields a McFadden R Sqaure o almost 0.94 yielding a very good fit. 
# We get to see Accuracy and Kappa values are high We shall do the prediction based on this model
carusage_pred<-predict.train(object = glm,cars_test[,regressors],type = "raw")
confusionMatrix(carusage_pred,cars_test[,outcome_variable], positive='1')

### Improving the model
# using glmnet method of caret package to try and run ridge Regression Model
library(caret)
trainctrlgn<-trainControl(method = 'cv',number = 10,returnResamp = 'none')
glmnet<-train(CarUsage~Age+Work.Exp+Salary+Distance+license, data = cars_SMOTE,
                  method = 'glmnet', trControl = trainctrlgn)
glmnet
varImp(object = glmnet)
plot(varImp(object = glmnet), main="Variable Importance using Post Ridge Regularization")
# license and Age are the most significant variables followed by distance

# Regularisation model
carusage_pred<-predict.train(object = glmnet,cars_test[,regressors],type = "raw")
confusionMatrix(carusage_pred,cars_test[,outcome_variable], positive='1')

######################################################################
## Bagging & Boosting
######################################################################
##Split the original base data into test and train samples again
cars_data_lda <- read.csv("C:/Users/DELL/Desktop/Akshay/Group Assignments/Group Assignment 7 ML/Cars.csv")
cars_data_lda$Gender<-as.factor(cars_data_lda$Gender)
cars_data_lda$Engineer<-as.factor(cars_data_lda$Engineer)
cars_data_lda$MBA<-as.factor(cars_data_lda$MBA)
cars_data_lda<-knnImputation(cars_data_lda)

set.seed(123)
cars_split_lda<-createDataPartition(cars_data_lda$Transport, p=0.7,list = FALSE,times = 1)
cars_train_lda<-cars_data_lda[cars_split_lda,]
cars_test_lda<-cars_data_lda[-cars_split_lda,]
cars_train_lda$license<-as.factor(cars_train_lda$license)
cars_test_lda$license<-as.factor(cars_test_lda$license)

cartrain_lda.car<-cars_train_lda[cars_train_lda$Transport %in% c("Car", "Public Transport"),]
cartrain_lda.twlr<-cars_train_lda[cars_train_lda$Transport %in% c("2Wheeler", "Public Transport"),]

cartrain_lda.car$Transport<-as.character(cartrain_lda.car$Transport)
cartrain_lda.car$Transport<-as.factor(cartrain_lda.car$Transport)
cartrain_lda.twlr$Transport<-as.character(cartrain_lda.twlr$Transport)
cartrain_lda.twlr$Transport<-as.factor(cartrain_lda.twlr$Transport)

prop.table(table(cartrain_lda.car$Transport))
prop.table(table(cartrain_lda.twlr$Transport))

car_lda_twlrsm <- SMOTE(Transport~., data = cartrain_lda.twlr, perc.over = 150, perc.under=200)
table(car_lda_twlrsm$Transport)

car_lda_carsm <- SMOTE(Transport~., data = cartrain_lda.car, perc.over = 175, perc.under=200)
table(car_lda_carsm$Transport)

car_lda<-car_lda_carsm[car_lda_carsm$Transport %in% c("Car"),]
cars_train_ldasm<-rbind(car_lda_twlrsm,car_lda)
str(cars_train_ldasm)

## boosting
boostcontrol <- trainControl(number=10)
xgbGrid <- expand.grid(
  eta = 0.3,
  max_depth = 1,
  nrounds = 50,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1, subsample = 1
)

cars_boosting <-  train(Transport ~ .,cars_train_ldasm,trControl = boostcontrol,
                  tuneGrid = xgbGrid,metric = "Accuracy",method = "xgbTree")
cars_boosting

## predict using the test dataset
predictions_boosting<-predict(cars_boosting,cars_test_lda)
confusionMatrix(predictions_boosting,cars_test_lda$Transport)

#############################################################################
## Naive Bayes
#############################################################################
library(e1071)
NB_Model=naiveBayes(cars_train$Transport ~., data=cars_train)
NB_Model
#Prediction on the test dataset
NB_Predictions=predict(NB_Model,cars_test)
table(NB_Predictions,cars_test$Transport)
# prediction for test sample
NB_Predictions=predict(NB_Model,cars_test)
NB_Predictions
summary(NB_Predictions)

###########################################################################
## KNN
###########################################################################
### KNN
# Normalize continuous variables
cars_data$Salary = log(cars_data$Salary)
cars_test$Salary = log(cars_test$Salary)

library(caret)
trControl <- trainControl(method  = "cv", number  = 10)
fit.knn <- train(cars_data$Transport ~ ., method = "knn", data = cars_data,
                 trControl  = trControl,
                 metric     = "Accuracy",
                 preProcess = c("center","scale"))
fit.knn
KNN_predictions = predict(fit.knn,cars_train)
table(KNN_predictions, cars_train$Transport)

KNN_predictions = predict(fit.knn,cars_test)
table(KNN_predictions, cars_test$Transport)
predict(fit.knn,cars_data)









