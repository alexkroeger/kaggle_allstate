
## Clear
rm(list=ls())

## libs on libs
library(parallel)
library(parallelsugar)
library(data.table)
library(xgboost)



## Read in data
train <- fread("../data/train.csv")
test <- fread("../data/test.csv")

## Uncomment for random sample
sample <- sample(1:nrow(train),8000)
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

save(test,file='../data/test')

# Free up mem
rm(test,data)
gc()

eval_mae <- function (yhat,dx) {
  y = getinfo(dx, "label")
  error <- mean(abs((exp(y)-200)-(exp(yhat)-200)))
  return(list(metric="error",value=error))
}

xgb_grid = expand.grid(
  eta = seq(0.01, 0.02, 0.01),
  max_depth = c(5, 6, 7)
)


xgb_list <- split(xgb_grid,seq(nrow(xgb_grid)))
starttime <- proc.time()
maeParams <- mclapply(xgb_list,function(params) {
  etaparam <- params$eta
  maxdepparam <- params$max_depth
  
  dx <-xgb.DMatrix(as.matrix(x),label=y)
  

  ## Train the model
  xgb.model <- xgboost::xgb.cv(data=dx,
                      eta=etaparam,
                      max_depth=maxdepparam,
                      objective='reg:linear',
                      nrounds=200,
                      nfold=10,
                      early.stop.round = 200,
                      feval=eval_mae,
                      maximize=FALSE,
                      verbose=FALSE)
  
  ## to avoid overfitting, use lowest CV within x sd's of the minimum
  x = 0
  min.index <- which(xgb.model$test.error.mean==min(xgb.model$test.error.mean))
  test.cv <- xgb.model$test.error.mean[min.index]+(x*xgb.model$test.error.std[min.index])
  best.n <- min(which(xgb.model$test.error.mean<=test.cv))
  
  return(c(test.cv, best.n, maxdepparam, etaparam))
},mc.cores=1)
elapsed <- proc.time() - starttime
print(elapsed)


maeParams <- data.frame(matrix(unlist(maeParams),nrow=length(maeParams),byrow=T))
save(maeParams,file='../output/maeParams')

## find model with best params
index <- which(maeParams[1,]==min(maeParams[1,]))
opt.n <- maeParams[2,index]
opt.maxdep <- maeParams[3,index]
opt.eta <- maeParams[4,index]

opt.model <- xgb.train(data=dx,
                       eta=opt.eta,
                       max_depth=opt.maxdep,
                       objective='reg:linear',
                       nrounds=opt.n)

load('../data/test')
dtest <- xgb.DMatrix(as.matrix(test))


## Predict for test set
yhat <- predict(opt.model,dtest)
yhat <- exp(yhat)-200

## Fill in submission template
submission_file <- "../sample_submission.csv"
submission <- fread(submission_file,colClasses=c("integer","numeric"))
submission$loss <- yhat

write.csv(submission,"../output/xgb.csv",row.names=FALSE)

elapsed <- proc.time() - starttime
print(elapsed)