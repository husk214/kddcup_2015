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

tsne2tr <- as.matrix(read.csv("../misc/tsne2_3_train.csv", header=TRUE))
tsne2te <- as.matrix(read.csv("../misc/tsne2_3_test.csv", header=TRUE))

tsne3tr <- as.matrix(read.csv("../misc/tsne3_3_train.csv", header=TRUE))
tsne3te <- as.matrix(read.csv("../misc/tsne3_3_test.csv", header=TRUE))


cotr <- cotr[,-1]
cote <- cote[,-1]

dtr <- dtr[,-1]
dte <- dte[,-1]

#best 6-3 depth = 4 0.8880

X <- as.matrix(read.csv("../dataset/train/3.csv", header=FALSE))

remove_feature <- c(9:24)
# remove_feature <- c(2,4,6,8,9:24, 34,36,38,40,42,44,46,48,50,52, 207:290,355:381,397:426)
# remove_feature <- c(282,396,297,281,389,183,341,119,296,10,19,383,177,99,198,284,20,278,130,337,178,347,437,174,316,344,342,289,310,54,293,290,203,138,276,264168,129,182,419,95,340,176,274,179,295,165,136,103,135,366,314,287,302330,180,421,418,339,181,369,291,164,166,39,370,142,157,152,315,147,298423,263,292,240,160,420,357,123,146,414,422,141,415,148,137,143,254,139163,236,167,161,158,416,372,132,155,162,397,417,140,367,300,244,248,371262,125,412,413,398,131,128,243,265,151,356,261,406,364,405,407,238,403251,411,124,355,400,242,399,404,410,241,401,237,126,154,328,40,245,253329,408,239,402,257,409,260,159,258,133,134,250,249,255,127,235,23,246156,256,359,221,361,259,247,153,24,363,252,233,234,362,358,219,360,210220,22,211,215,63,218,21,228,222,223,56,217,232,227,226,229,230,225,71,231,212,86,216,214,224,213,208,207,58,209,68,67,74,75,77,72,69,87,64,65,76,89,80,88,73,57,93,81,79,78,70,92,91,84,62,439,83,66,59,94,85,82,60,90,61,442)

# remove_feature <- c(207:290,355:381,397:426, 442)
y <- X[,ncol(X)]
tmpx <- X[,-ncol(X)]

tmpx <- cbind(tmpx, cotr, dtr)
# tmpx <- cbind(tmpx, cotr, dtr,tsne2tr)

tmpx <- tmpx[,-remove_feature]

x <- standardize(tmpx, get_std_coeff(tmpx))
xx <- cbind(x,rep(1,nrow(x)))
xgmat <- xgb.DMatrix(data=xx, label=y)

# ss <- sample(nrow(xx),70000)
# subxx <- xx[ss,]
# suby <- y[ss]
# subtxx <- xx[-ss,]
# subty <- y[-ss]

# subxgmat <- xgb.DMatrix(data=subxx, label=suby)
# subtxgmat <- xgb.DMatrix(data=subtxx, label=subty)
# watchlist <- list(train=subxgmat, test=subtxgmat)

# bst <- xgb.train(data=subxgmat, max.depth=3, eta=0.1, nthread = 4, nround=300, watchlist=watchlist, objective = "binary:logistic", eval.metric="auc")

vx <-as.matrix( read.csv("../dataset/test/1-1.csv", header=FALSE))

eid <- vx[,ncol(vx)]
tmpvx <- vx[,-ncol(vx)]
tmpvx <- tmpvx[,-remove_feature]
tmpvx <- cbind(tmpvx, cote, dte)
# tmpvx <- cbind(tmpvx, cote, dte,knn9te)

vx <- standardize(tmpvx, get_std_coeff(tmpx))
vxx <-cbind(vx,rep(1,nrow(vx)))

outlier <- as.matrix( read.csv("../dataset/knn20_outlier.csv", header=T) )
num_removal <- floor(nrow(outlier) * 0.03)
removal_instance <- outlier[1:num_removal, 1]

oxx <- xx[-removal_instance, ]
oy <- y[-removal_instance ]
oxgmat <- xgb.DMatrix(data=oxx, label=oy)


param <- list("objective"="binary:logistic", "eval_metric"="auc",  "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1,"max_depth"=3)

bst.cv <- xgb.cv(param=param, data=oxgmat, nfold=10, nrounds=500)


bst <- xgboost(param=param, data=oxgmat, nrounds=200)

imp <- xgb.importance(model=bst)
impfe <- as.vector(as.numeric((imp$Feature)))
impfe <- impfe +1
impga <- as.vector(as.numeric(imp$Gain))
impvec <- cbind(impfe, impga)
print(imp)
xgb.plot.importance(imp)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)


param <- list("objective"="binary:logistic", "eval_metric"="auc",  "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1,"max_depth"=3)
bst <- xgboost(param=param, data=oxgmat, nrounds=180)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/olxgb_3-3.csv", sep=',', quote=FALSE,row.names=FALSE, col.names=FALSE)

param <- list("objective"="binary:logistic", "eval_metric"="auc",  "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1,"max_depth"=4)
bst <- xgboost(param=param, data=oxgmat, nrounds=90)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/olxgb_3-4.csv", sep=',', quote=FALSE,row.names=FALSE, col.names=FALSE)

param <- list("objective"="binary:logistic", "eval_metric"="auc",  "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1,"max_depth"=5)
bst <- xgboost(param=param, data=oxgmat, nrounds=55)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/olxgb_3-5.csv", sep=',', quote=FALSE,row.names=FALSE, col.names=FALSE)

param <- list("objective"="binary:logistic", "eval_metric"="auc",  "nthread"=8, "bst:eta" =0.2, "bst:gamma"=0, "min.child.weight"=1,"max_depth"=6)
bst <- xgboost(param=param, data=oxgmat, nrounds=40)
pred <- predict(bst, vxx)
output <- cbind(eid, pred)
write.table(output, file="../submission/olxgb_3-6.csv", sep=',', quote=FALSE,row.names=FALSE, col.names=FALSE)

rx <- as.data.frame(cbind(tmpx, y))
rx$y <- as.factor(rx$y)
# forest <- randomForest(y~.,data=rx, mtry=2, auto.weight=TRUE)
# print(importance(forest))

sabsam <- sample(nrow(tmpx), 30000)
subrx <- rx[sabsam, ]
forest <- randomForest(y~.,data=subrx, mtry=2, auto.weight=TRUE)
impvar <- order(forest$importance,decreasing=T)
print(importance(forest))

#q(save="no");
