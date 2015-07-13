library(xgboost)
library(Matrix)
library(AUC)
library(LiblineaR)
library(randomForest)


get_std_coeff <- function(X)
{
  std_ <- apply(X, 2, sd)
  std_[std_ < 1e-16] <- 1.0
  return ( list(apply(X, 2, mean), std_) )
}

standardize <- function(X, std_coeff)
{
  X <- scale(X, std_coeff[[1]], std_coeff[[2]])
  return (X)
}

cotr <- as.matrix(read.csv("../dataset/complete_lecture_train.csv", header=TRUE))
cote <- as.matrix(read.csv("../dataset/complete_lecture_test.csv", header=TRUE))

dtr <- as.matrix(read.csv("../dataset/numbers_depth.csv", header=TRUE))
dte <- as.matrix(read.csv("../dataset/numbers_depth_test.csv", header=TRUE))

cotr <- cotr[,-1]
cote <- cote[,-1]

dtr <- dtr[,-1]
dte <- dte[,-1]


X <- as.matrix(read.csv("../dataset/train/3.csv", header=FALSE))

remove_feature <- c(1,3,5,7,9:24,33,35,37,39,41,43,45,47,56:94)
y <- X[,ncol(X)]
tmpx <- X[,-ncol(X)]
tmpx <- tmpx[,-remove_feature]

nX <- as.matrix(read.csv("../submission/newclassifydata702i.csv", header=TRUE))
nx <- nX[,c(-1,-ncol(nX))]
tmpx <- cbind(tmpx,nx)

x <- standardize(tmpx, get_std_coeff(tmpx))
xx <- cbind(x,rep(1,nrow(x)))
xgmat <- xgb.DMatrix(data=xx, label=y)

param <- list("objective"="binary:logistic", "eval_metric"="auc", "nthread"=16, "bst:eta" =0.075, "bst:gamma"=0.05, "min.child.weight"=1,  "max_depth"=4)

bst.cv <- xgb.cv(param=param, data=xgmat, nfold=10, nrounds=500)


bst <- xgboost(param=param, data=xgmat, nrounds=110)


vx <-as.matrix( read.csv("../dataset/test/3.csv", header=FALSE))
eid <- vx[,ncol(vx)]
tmpvx <- vx[,-ncol(vx)]
tmpvx <- tmpvx[,-remove_feature]
nvX <- as.matrix(read.csv("../submission/testclassifydata702i.csv", header=TRUE))
nvx <- nvX[,-1]
tmpvx <- cbind(tmpvx,nvx)

vx <- standardize(tmpvx, get_std_coeff(tmpx))
vxx <-cbind(vx,rep(1,nrow(vx)))


param <- list("objective"="binary:logistic", "eval_metric"="auc", "nthread"=16, "bst:eta" =0.075, "bst:gamma"=0.05, "min.child.weight"=1,  "max_depth"=3)
bst <- xgboost(param=param, data=xgmat, nrounds=430)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/xgb_3311-3.csv", sep=',', quote=F, row.names=F,col.names=F)


param <- list("objective"="binary:logistic", "eval_metric"="auc", "nthread"=16, "bst:eta" =0.075, "bst:gamma"=0.05, "min.child.weight"=1,  "max_depth"=4)
bst <- xgboost(param=param, data=xgmat, nrounds=300)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/xgb_3311-4.csv", sep=',', quote=F, row.names=F,col.names=F)

param <- list("objective"="binary:logistic", "eval_metric"="auc", "nthread"=16, "bst:eta" =0.075, "bst:gamma"=0.05, "min.child.weight"=1,  "max_depth"=5)
bst <- xgboost(param=param, data=xgmat, nrounds=170)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/xgb_3311-5.csv", sep=',', quote=F, row.names=F,col.names=F)

param <- list("objective"="binary:logistic", "eval_metric"="auc", "nthread"=16, "bst:eta" =0.075, "bst:gamma"=0.05, "min.child.weight"=1,  "max_depth"=6)
bst <- xgboost(param=param, data=xgmat, nrounds=160)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/xgb_3311-6.csv", sep=',', quote=F, row.names=F,col.names=F)

#q(save="no");
