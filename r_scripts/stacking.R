library(xgboost)
library(Matrix)
library(LiblineaR)
library(randomForest)
library(Rtsne)
library(FNN)
options( java.parameters = "-Xmx8g" )
library(extraTrees)
library(rpart)
library(adabag)
library(kernlab)
library(AUC)

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

test_subx <- x[cv_fold[[6]]$test,]
test_subyy <- y[cv_fold[[6]]$test]
test_suby <- factor(test_subyy)

for (index in 1:5) {
  subx <- x[cv_fold[[index]]$test,]
  subyy <- y[cv_fold[[index]]$test]
  suby <- factor(subyy)

  # extraTrees
  for (numTrees in c(2,4,6,8)) {
    et <- extraTrees(subx, suby, mtry=numTrees, numThreads=8)
    etprob <- predict(et, x, probability=T)
    output_fn <- paste("../dataset/stacked2/et",numTrees, "_",index,"_train.csv", sep="")
    write.csv(etprob[,2], output_fn, quote=FALSE,row.names=FALSE)
    output_fn <- paste("../dataset/stacked2/et",numTrees, "_",index,"_test.csv", sep="")
    etprob <- predict(et, vx, probability=T)
    write.csv(etprob[,2],output_fn, quote=FALSE,row.names=FALSE)
  }

  # kNN
  for (nnofk in c(3,5,9,21)) {
    fnnout <- FNN::knn(train=subx, test=x, cl=suby, k=nnofk,prob=T, algorithm="kd_tree")
    hatclass <- as.vector(as.numeric(fnnout)) -1
    output_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_train.csv", sep="")
    write.csv(hatclass, output_fn, quote=FALSE,row.names=FALSE)
    prob_dropout <- abs(attr(fnnout, "prob") - abs(as.vector(as.numeric(fnnout)) -2))
    output_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_prob_train.csv", sep="")
    write.csv(prob_dropout, output_fn, quote=FALSE,row.names=FALSE)

    fnnout <- FNN::knn(train=subx, test=vx, cl=suby, k=nnofk,prob=T, algorithm="kd_tree")
    hatclass <- as.vector(as.numeric(fnnout)) -1
    output_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_test.csv", sep="")
    write.csv(hatclass, output_fn, quote=FALSE,row.names=FALSE)
    prob_dropout <- abs(attr(fnnout, "prob") - abs(as.vector(as.numeric(fnnout)) -2))
    output_fn <- paste("../dataset/stacked2/knn",nnofk, "_",index,"_prob_test.csv", sep="")
    write.csv(prob_dropout, output_fn, quote=FALSE,row.names=FALSE)
  }

  # liblinear
  for (modelnumber in c(0,2,5)) {
    if (modelnumber==5) {
      m <- LiblineaR(data=subx, target=suby, type=modelnumber, cost=0.003,epsilon=0.001, bias=TRUE)
    } else {
      m <- LiblineaR(data=subx, target=suby, type=modelnumber, cost=0.04,epsilon=0.001, bias=TRUE)
    }
    p <- predict(m, x, decisionValues=TRUE)
    pp <- p$decisionValue
    pp  <- pp[,-ncol(pp)]
    output_fn <- paste("../dataset/stacked2/liblinear",modelnumber, "_",index,"_train.csv", sep="")
    write.csv(pp, output_fn, quote=FALSE,row.names=FALSE)
    p <- predict(m, vx, decisionValues=TRUE)
    pp <- p$decisionValue
    pp <- pp[,-ncol(pp)]
    output_fn <- paste("../dataset/stacked2/liblinear",modelnumber, "_",index,"_test.csv", sep="")
    write.csv(pp, output_fn, quote=FALSE,row.names=FALSE)
  }

  # xgboost
  subxx <- cbind(subx,rep(1,nrow(subx)))
  subxgmat <- xgb.DMatrix(data=subxx, label=subyy)
  for (max_d in c(3,4)) {
    param <- list("objective"="binary:logistic", "eval_metric"="auc", "max_depth"=max_d, "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1)
    bst <- xgboost(param=param, data=subxgmat, nrounds=110)
    pred <- predict(bst, xx)
    output_fn <- paste("../dataset/stacked2/xgb",max_d, "_",index,"_train.csv", sep="")
    write.csv(pred, output_fn, quote=FALSE,row.names=FALSE)
    pred <- predict(bst, vxx)
    output_fn <- paste("../dataset/stacked2/xgb",max_d, "_",index,"_test.csv", sep="")
    write.csv(pred, output_fn, quote=FALSE,row.names=FALSE)
  }
  for (max_d in c(5,6)) {
    param <- list("objective"="binary:logistic", "eval_metric"="auc", "max_depth"=max_d, "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1)
    bst <- xgboost(param=param, data=subxgmat, nrounds=50)
    pred <- predict(bst, xx)
    output_fn <- paste("../dataset/stacked2/xgb",max_d, "_",index,"_train.csv", sep="")
    write.csv(pred, output_fn, quote=FALSE,row.names=FALSE)
    pred <- predict(bst, vxx)
    output_fn <- paste("../dataset/stacked2/xgb",max_d, "_",index,"_test.csv", sep="")
    write.csv(pred, output_fn, quote=FALSE,row.names=FALSE)
  }

  # #randomForest
  subrx <- as.data.frame(cbind(subx,suby))
  subrx$suby <- as.factor(subrx$suby)
  for (numtrees in c(2,4,6,8)) {
    forest <- randomForest(suby~.,data=subrx, mtry=2, auto.weight=TRUE)
    prob <- predict(forest, x, type="prob")
    pp <- prob[,2]
    output_fn <- paste("../dataset/stacked2/rf",numtrees, "_",index,"_train.csv", sep="")
    write.csv(pp, output_fn, quote=FALSE,row.names=FALSE)
    prob <- predict(forest, vx, type="prob")
    pp <- prob[,2]
    output_fn <- paste("../dataset/stacked2/rf",numtrees, "_",index,"_test.csv", sep="")
    write.csv(pp, output_fn, quote=FALSE,row.names=FALSE)
  }

  # kernlab
  # for (powi in c(-5:2)) {
  #   c <- 2^powi
  #   model <- ksvm(suby~.
  #     ,data =subrx,
  #     type = "C-svc",
  #     kernel = "rbfdot",
  #     C =c,
  #     prob.model = TRUE
  #   )
  #   fx <- predict(model,x,type="decision")
  #   output_fn <- paste("../dataset/stacked2/rbfsvm",powi, "_",index,"_train.csv", sep="")
  #   write.csv(fx, output_fn, quote=FALSE,row.names=FALSE)
  #   fx <- predict(model,vx,type="decision")
  #   output_fn <- paste("../dataset/stacked2/rbfsvm",powi, "_",index,"_test.csv", sep="")
  #   write.csv(fx, output_fn, quote=FALSE,row.names=FALSE)
  #   fx <- predict(model,test_subx,type="decision")
  #   sauc <- auc(roc(fx,factor(test_subyy)))
  #   ppos <- paste("rbfsvm c=",c, index ,", auc = ", sauc )
  #   print(ppos)
  # }

}

q(save="no");
