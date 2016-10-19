## Clear
rm(list=ls())


## libs on libs
library(randomForest)
library(data.table)
library(gbm)


## Read in training dataset
train <- fread("../data/train.csv")
names(train)
submission_file <- "../sample_submission.csv"

## Convert character to freakin' factor variables
# list of character vars
cat_var <- names(train)[which(sapply(train, is.character))]
train[,(cat_var):=lapply(.SD, as.factor),.SDcols=cat_var]
cont_var <- names(train)[which(sapply(train,is.numeric))]

rm(cat_var,cont_var)

## Uncomment for random sample
#sample <- sample(1:nrow(train),500)
#train <- train[sample,]

## Run boosting model
## CV error from dif. models
gbmcv <- function(list) {
  shrink=list[[1]]
  depth=list[[2]]
  gbm.model <- gbm(loss~.-id,data=train,shrinkage=shrink,distribution="gaussian",n.trees=5000,cv.folds=10,interaction.depth=depth)
  c(shrink,depth,min(gbm.model$cv.error),which.min(gbm.model$cv.error))
} 

## Vector of shrinkage params
num_depths <- 4
max_shrink <- 0.02
shrink.params <- sort(rep(seq(0.001,max_shrink,0.001),times=num_depths))
depths <- rep(seq(1,num_depths,1),length(shrink.params)/num_depths)

params <- cbind(shrink.params,depths)
list <- split(params,seq(nrow(params)))

#result = apply(X=params,MARGIN=1,FUN=gbmcv)
result = sapply(list,gbmcv)
result = t(result)

best.shrink = result[which.min(result[,3]),1]
best.depth = result[which.min(result[,3]),2]

gbm.model <- gbm(loss~.-id,data=train,shrinkage=best.shrink,
                 distribution="gaussian",n.trees=10000,
                 cv.folds=10,interaction.depth=best.depth)

best.ntree <- which.min(gbm.model$cv.error)

## Predict test set with boosting
test <- fread("../data/test.csv")

# Again, convert character to factor
cat_var <- names(test)[which(sapply(test, is.character))]
test[,(cat_var):=lapply(.SD, as.factor),.SDcols=cat_var]
yhat <- predict(gbm.model,newdata=test,n.trees=best.ntree)

rm(cat_var)

submission_file <- "../sample_submission.csv"
submission <- fread(submission_file,colClasses=c("integer","numeric"))
submission$loss <- yhat

write.csv(submission,"../output/outboost_optimized.csv",row.names=FALSE)
