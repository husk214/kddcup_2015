options( java.parameters = "-Xmx8g" )
library(extraTrees)
library(caret)
library(AUC)

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

set.seed(777)

X <- as.matrix(read.csv("../dataset/train/3.csv", header=FALSE))

remove_feature <- c(1,3,5,7,9:24,33,35,37,39,41,43,45,47,56:94)
y <- X[,ncol(X)]
tmpx <- X[,-ncol(X)]
tmpx <- tmpx[,-remove_feature]

nX <- as.matrix(read.csv("../submission/newclassifydata702i.csv", header=TRUE))
nx <- nX[,c(-1,-ncol(nX))]
tmpx <- cbind(tmpx,nx)

x <- standardize(tmpx, get_std_coeff(tmpx))


cv_fold <- f_K_fold(nrow(X), 6)

trainx <- x[cv_fold[[6]]$train,]
trainy <- factor(y[cv_fold[[6]]$train])
testx <- x[cv_fold[[6]]$test,]
testy <- factor(y[cv_fold[[6]]$test])

et <- extraTrees(trainx, trainy, numThreads=8, mtry=2, ntree=300 )
etprob <- predict(et, testx, probability=T)
auc(roc(etprob[,2], factor(testy) ) )
