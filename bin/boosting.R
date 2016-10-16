
## Clear
rm(list=ls())


## libs on libs
library(randomForest)
library(data.table)
library(gbm)

## Read in training dataset
train <- fread("../data/train.csv")
names(train)

## Convert character to freakin' factor variables
# list of character vars
cat_var <- names(train)[which(sapply(train, is.character))]
train[,(cat_var):=lapply(.SD, as.factor),.SDcols=cat_var]
cont_var <- names(train)[which(sapply(train,is.numeric))]

rm(cat_var,cont_var)

## Run boosting model
boost.model <- gbm(loss~.-id,data=train,distribution="gaussian",n.trees=5000,interaction.depth = 1)

## Predict test set with boosting
test <- fread("../data/test.csv")

# Again, convert character to factor
cat_var <- names(test)[which(sapply(test, is.character))]
test[,(cat_var):=lapply(.SD, as.factor),.SDcols=cat_var]
yhat <- predict(boost.model,newdata=test,n.trees=5000)

rm(cat_var)

outboost <- as.data.frame(cbind(test$id,yhat))

outnames <- c("id","loss")
names(outboost) <- outnames

write.table(outboost,"../output/outboost.csv",sep=",",row.names=FALSE)
