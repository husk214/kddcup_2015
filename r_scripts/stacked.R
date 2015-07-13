library(xgboost)
library(Matrix)
library(LiblineaR)
library(randomForest)
library(Rtsne)
library(FNN)
options( java.parameters = "-Xmx8g" )
library(extraTrees)

set.seed(777)

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

f_K_fold <- function(Nobs,K=6){
    rs <- runif(Nobs)
    id <- seq(Nobs)[order(rs)]
    k <- as.integer(Nobs*seq(1,K-1)/K)
    k <- matrix(c(0,rep(k,each=2),Nobs),ncol=2,byrow=TRUE)
    k[,1] <- k[,1]+1
    l <- lapply(seq.int(K),function(x,k,d)
                list(train=d[!(seq(d) %in% seq(k[x,1],k[x,2]))],
                     test=d[seq(k[x,1],k[x,2])]),k=k,d=id)
   return(l)
}

cotr <- as.matrix(read.csv("../dataset/complete_lecture_train.csv", header=TRUE))
cote <- as.matrix(read.csv("../dataset/complete_lecture_test.csv", header=TRUE))

dtr <- as.matrix(read.csv("../dataset/numbers_depth.csv", header=TRUE))
dte <- as.matrix(read.csv("../dataset/numbers_depth_test.csv", header=TRUE))

cotr <- cotr[,-1]
cote <- cote[,-1]

dtr <- dtr[,-1]
dte <- dte[,-1]
colnames(dte)[6] <- "depth."

X <- as.matrix(read.csv("../dataset/train/3-1.csv", header=FALSE))

cv_fold <- f_K_fold(nrow(X), 6)

course_feature <- c(56:94)
time_feature <- c(9:24)
row_feature <- c(1,3,5,7,9,11,13,15,17,19,21,23,33,35,37,39,41,43,45,47,49,51)
row_feature1 <- c(1,3,5,7,33,35,37,39,41,43,45,47,49,51)
log_feature <- c(2,4,6,8,10,12,14,16,18,20,22,24,34,36,38,40,42,44,46,48,50,52)
log_feature1 <- c(2,4,6,8,34,36,38,40,42,44,46,48,50,52)
ma_feature <- c(157:380)
ma_feature1 <- c(269:352)
must_remove_feature <- c(382,387)

remove_feature <- c(row_feature1,time_feature,must_remove_feature)

tmpx <- X[,-ncol(X)]
tmpx <- cbind(tmpx, cotr)
tmpx <- cbind(tmpx, dtr)
tmpx <- tmpx[,-remove_feature]

x <- standardize(tmpx, get_std_coeff(tmpx))
xx <- cbind(x,rep(1,nrow(x)))

vx <-as.matrix( read.csv("../dataset/test/3-1.csv", header=FALSE))

eid <- vx[,ncol(vx)]
tmpvx <- vx[,-ncol(vx)]
tmpvx <- cbind(tmpvx, cote)
tmpvx <- cbind(tmpvx, dte)
tmpvx <- tmpvx[,-remove_feature]
vx <- standardize(tmpvx, get_std_coeff(tmpx))
vxx <-cbind(vx,rep(1,nrow(vx)))


y <- X[,ncol(X)]
fy <- factor(y)

sx <- c(rep(1:nrow(X)))
vsx <- c(rep(1:nrow(vx)))

for (index in 1:5) {

  # extraTrees
  for (numTrees in c(2)) {
    input_fn <- paste("../dataset/stacked2/et",numTrees, "_",index,"_train.csv", sep="")
    sx <- cbind(read.csv(input_fn, header=TRUE), sx)
    input_fn <- paste("../dataset/stacked2/et",numTrees, "_",index,"_test.csv", sep="")
    vsx <- cbind(read.csv(input_fn, header=TRUE), vsx)
  }

  # kNN
  for (nnofk in c(9)) {
    input_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_train.csv", sep="")
    sx <- cbind(read.csv(input_fn, header=TRUE), sx)
    input_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_prob_train.csv", sep="")
    sx <- cbind(read.csv(input_fn, header=TRUE), sx)

    input_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_test.csv", sep="")
    vsx <- cbind(read.csv(input_fn, header=TRUE), vsx)
    input_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_prob_test.csv", sep="")
    vsx <- cbind(read.csv(input_fn, header=TRUE), vsx)
  }

  # liblinear
  for (modelnumber in c(0,2,5)) {
    input_fn <- paste("../dataset/stacked2/liblinear",modelnumber, "_",index,"_train.csv", sep="")
    sx <- cbind(read.csv(input_fn, header=TRUE), sx)

    input_fn <- paste("../dataset/stacked2/liblinear",modelnumber, "_",index,"_test.csv", sep="")
    vsx <- cbind(read.csv(input_fn, header=TRUE), vsx)
  }

  # xgboost
  for (max_d in c(3,4,5,6)) {
    input_fn <- paste("../dataset/stacked2/xgb",max_d, "_",index,"_train.csv", sep="")
    sx <- cbind(read.csv(input_fn, header=TRUE), sx)
    input_fn <- paste("../dataset/stacked2/xgb",max_d, "_",index,"_test.csv", sep="")
    vsx <- cbind(read.csv(input_fn, header=TRUE), vsx)
  }

  #randomForest
  for (numtrees in c(2)) {
    input_fn <- paste("../dataset/stacked2/rf",numtrees, "_",index,"_train.csv", sep="")
    sx <- cbind(read.csv(input_fn, header=TRUE), sx)
    input_fn <- paste("../dataset/stacked2/rf",numtrees, "_",index,"_test.csv", sep="")
    vsx <- cbind(read.csv(input_fn, header=TRUE), vsx)
  }
}

sx <- as.matrix(sx)
vsx <- as.matrix(vsx)

train_sx <- sx[cv_fold[[6]]$train, ]
train_sy <- y[cv_fold[[6]]$train]
test_sx <- sx[cv_fold[[6]]$test, ]
test_sy <- y[cv_fold[[6]]$test]

xgmat <- xgb.DMatrix(data=train_sx, label=train_sy)
vxgmat <- xgb.DMatrix(data=test_sx, label=test_sy)
watchlist <- list(train=xgmat, test=vxgmat)
param <- list("objective"="binary:logistic", "eval_metric"="auc", "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1,"max_depth"=3)
bst <- xgb.train(data=xgmat, param=param, nround=100, watchlist=watchlist, objective = "binary:logistic", eval.metric="auc")


for (cc in rep(-10:10)) {
  c <- 2^cc
  wi <- as.vector(c(c,4*c))
  names(wi) <- c("1","0")
  m <- LiblineaR(data=train_sx, target=train_sy, type=0,wi=wi, cost=c,epsilon=0.001, bias=TRUE)
  # m <- LiblineaR(data=train_sx, target=train_sy, type=0,wi=NULL, cost=0.04,epsilon=0.001, bias=TRUE)
  p <- predict(m, test_sx, decisionValues=TRUE)
  pp <- p$decisionValue
  pp  <- pp[,-ncol(pp)]
  sauc <- auc(roc(pp,factor(test_sy)))
  ppos <- paste("linear l2r lr c=",c ,", auc = ", sauc )

  p <- predict(m, train_sx, decisionValues=TRUE)
  pp <- p$decisionValue
  pp  <- pp[,-ncol(pp)]
  sauc <- auc(roc(pp,factor(train_sy)))
  ppos <- paste(ppos, ", train aus =", sauc )
  print(ppos)
  print(m)
}
q(save="no");
