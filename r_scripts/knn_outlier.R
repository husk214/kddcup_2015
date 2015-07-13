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

train_eid <- as.matrix(read.csv("../dataset/train/enrollment_train.csv", header=TRUE))
train_eid <- as.numeric(train_eid[,-c(2,3)])

cotr <- cotr[,-1]
cote <- cote[,-1]

dtr <- dtr[,-1]
dte <- dte[,-1]

X <- as.matrix(read.csv("../dataset/train/1-1.csv", header=FALSE))

remove_feature <- c(207:290,355:381,397:426)
y <- X[,ncol(X)]
tmpx <- X[,-ncol(X)]

tmpx <- cbind(tmpx, cotr, dtr)

tmpx <- tmpx[,-remove_feature]

x <- standardize(tmpx, get_std_coeff(tmpx))

fnnout <- FNN::knn(train=x, test=x, cl=y, k=20,prob=T, algorithm="kd_tree")
prob_dropout <- abs(attr(fnnout, "prob") - abs(as.vector(as.numeric(fnnout)) -2))
miss_rate <- abs(prob_dropout - y)
outlier <- as.data.frame(cbind(train_eid, miss_rate))
sorted_outlier <- outlier[order(miss_rate),]
write.csv(sorted_outlier, "../dataset/knn20_outlier.csv", quote=FALSE,row.names=FALSE)
