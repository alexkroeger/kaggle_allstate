## Clear
rm(list=ls())

## libs on libs
library(data.table)
library(xgboost)

## Read in data
train <- fread("../data/train.csv")
test <- fread("../data/test.csv")

## Uncomment for random sample
sample <- sample(1:nrow(train),70000)
train <- train[sample,]


## response = loss
## Use log of the translated response
y <-log(train$loss+200)

## Take out ID, loss
train <- subset(train,select=-c(id,loss))
test <- subset(test,select=-c(id))

## combine
trainset = nrow(train)
data = rbind(train,test)
rm(train,test)

## make the factor vars numeric
predictors = names(data)

for (p in predictors) {
  if (class(data[[p]])=="character") {
    levels <- unique(data[[p]])
    data[[p]] <- as.integer(factor(data[[p]], levels=levels))
  }
}

## Make XGB objects
x <- data[1:trainset,]
test <- data[(trainset+1):nrow(data),]

dx <-xgb.DMatrix(as.matrix(x),label=y)
dtest <- xgb.DMatrix(as.matrix(test))

#save(dtest,file='../data/dtest')

eval_mae <- function (yhat,dx) {
  y = getinfo(dx, "label")
  error <- mean(abs((exp(y)-200)-(exp(yhat)-200)))
  return(list(metric="error",value=error))
}

## list of xgb parameters
params <- list(objective='reg:linear',
               max_depth=6)

## Train the model
xgb.model <- xgb.cv(params,
                    data=dx,
                    nrounds=1000,
                    nfold=10,
                    early.stop.round = 20,
                    feval=eval_mae,
                    maximize=FALSE)

best_n <- which(xgb.model$test.error.mean==min(xgb.model$test.error.mean))

opt.model <- xgb.train(params,
                       dx,
                       nrounds=best_n)

## Predict for test set
yhat <- predict(opt.model,dtest)
yhat <- exp(yhat)-200

## Fill in submission template
submission_file <- "../sample_submission.csv"
submission <- fread(submission_file,colClasses=c("integer","numeric"))
submission$loss <- yhat

write.csv(submission,"../output/xgb.csv",row.names=FALSE)