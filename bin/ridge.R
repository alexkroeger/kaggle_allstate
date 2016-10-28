## Clear
rm(list=ls())

## libs on libs
library(data.table)
library(glmnet)

## Read in training dataset
train <- fread("../data/train.csv")
test <- fread("../data/test.csv")

## Uncomment for random sample
#sample <- sample(1:nrow(train),50000)
#train <- train[sample,]

test$loss <- 0
test$test <- 1
train$test <- 0

data <- rbind(test,train)
testset <- which(data$test == 1)
trainset <- which(data$test == 0)

rm(test,train)

## Matrix for regression
X <- model.matrix(loss~.-id-test,data)

## Take log of loss
Y <- log(data$loss[trainset])

# get rid of original training set to save space
rm(data)

## Train using ridge regression
fit.ridge <- cv.glmnet(X[trainset,],Y,alpha=0)

opt.lambda = fit.ridge$lambda.min

## Predict for test set
X.test <- X[testset,]
rm(X)
yhat <- predict(fit.ridge,s=opt.lambda,newx=X.test)
yhat <- exp(yhat)

## Fill in submission template
submission_file <- "../sample_submission.csv"
submission <- fread(submission_file,colClasses=c("integer","numeric"))
submission$loss <- yhat

write.csv(submission,"../output/ridge_optimized.csv",row.names=FALSE)
