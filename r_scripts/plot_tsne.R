require(ggplot2)
set.seed(77)
tsne2 <- as.matrix(read.csv("../misc/tsne2_3_train.csv", header=T ))
y <- read.csv("../misc/truth_train.csv", header = F)
y <- y[,2]
tsne2 <- as.data.frame(cbind(tsne2, y))
tsne2[,3] <- as.factor(tsne2[,3])
tsne2 <- tsne2[sample(nrow(tsne2), 5000),]
colnames(tsne2) <- c("x", "y", "dropout")
gp = ggplot(tsne2, aes(x=x,y=y)) + geom_point(aes(colour=dropout), alpha=0.7)
print(gp)
ggsave(file = "../figure/tsne2_3.png", plot = gp)

set.seed(77)
tsne2 <- as.matrix(read.csv("../misc/tsne2_2_train.csv", header=T ))
y <- read.csv("../misc/truth_train.csv", header = F)
y <- y[,2]
tsne2 <- as.data.frame(cbind(tsne2, y))
tsne2[,3] <- as.factor(tsne2[,3])
tsne2 <- tsne2[sample(nrow(tsne2), 5000),]
colnames(tsne2) <- c("x", "y", "dropout")
gp = ggplot(tsne2, aes(x=x,y=y)) + geom_point(aes(colour=dropout), alpha=0.7)
print(gp)
ggsave(file = "../figure/tsne2_2.png", plot = gp)


set.seed(77)
tsne2 <- as.matrix(read.csv("../misc/tsne2_1_train.csv", header=T ))
y <- read.csv("../misc/truth_train.csv", header = F)
y <- y[,2]
tsne2 <- as.data.frame(cbind(tsne2, y))
tsne2[,3] <- as.factor(tsne2[,3])
tsne2 <- tsne2[sample(nrow(tsne2), 5000),]
colnames(tsne2) <- c("x", "y", "dropout")

gp = ggplot(tsne2, aes(x=x,y=y)) + geom_point(aes(colour=dropout), alpha=0.7)
print(gp)
ggsave(file = "../figure/tsne2_1.png", plot = gp)



library(rgl)

set.seed(77)
tsne3 <- as.matrix(read.csv("../misc/tsne3_2_train.csv", header=T ))
y <- read.csv("../misc/truth_train.csv", header = F)
y <- y[,2]
y0 <- which(y==0)
y1 <- which(y==1)
tsne3_y0 <- as.data.frame(cbind(tsne3[y0,], y[y0]))
tsne3_y0 <- tsne3_y0[sample(nrow(tsne3_y0), 200),]
tsne3_y1 <- as.data.frame(cbind(tsne3[y1,], y[y1]))
tsne3_y1 <- tsne3_y1[sample(nrow(tsne3_y1), 800),]
colnames(tsne3_y0) <- c("x", "y", "Not dropout")
colnames(tsne3_y1) <- c("x", "y", "dropout")
col_ndrop <- rgb(0.988, 0.459, 0.439)
col_drop <- rgb(0, 0.757, 0.769)
points3d(tsne3_y0, col=col_ndrop, size=5)
points3d(tsne3_y1, col=col_drop, size=5)
axes3d()
legend3d("topright", c("not dropout", "dropout"),  pch = 16, col=c(col_ndrop,col_drop),cex=1, inset=c(0.02))
writeWebGL(dir="../figure/tsne3_2",width=1000, height=1000)
rgl.close()

set.seed(77)
tsne3 <- as.matrix(read.csv("../misc/tsne3_1_train.csv", header=T ))
y <- read.csv("../misc/truth_train.csv", header = F)
y <- y[,2]
y0 <- which(y==0)
y1 <- which(y==1)
tsne3_y0 <- as.data.frame(cbind(tsne3[y0,], y[y0]))
tsne3_y0 <- tsne3_y0[sample(nrow(tsne3_y0), 200),]
tsne3_y1 <- as.data.frame(cbind(tsne3[y1,], y[y1]))
tsne3_y1 <- tsne3_y1[sample(nrow(tsne3_y1), 800),]
colnames(tsne3_y0) <- c("x", "y", "Not dropout")
colnames(tsne3_y1) <- c("x", "y", "dropout")
col_ndrop <- rgb(0.988, 0.459, 0.439)
col_drop <- rgb(0, 0.757, 0.769)
points3d(tsne3_y0, col=col_ndrop, size=5)
points3d(tsne3_y1, col=col_drop, size=5)
axes3d()
legend3d("topright", c("not dropout", "dropout"),  pch = 16, col=c(col_ndrop,col_drop),cex=1, inset=c(0.02))
writeWebGL(dir="../figure/tsne3_1",width=1000, height=1000)
rgl.close()


set.seed(77)
tsne3 <- as.matrix(read.csv("../misc/tsne3_3_train.csv", header=T ))
y <- read.csv("../misc/truth_train.csv", header = F)
y <- y[,2]
y0 <- which(y==0)
y1 <- which(y==1)
tsne3_y0 <- as.data.frame(cbind(tsne3[y0,], y[y0]))
tsne3_y0 <- tsne3_y0[sample(nrow(tsne3_y0), 200),]
tsne3_y1 <- as.data.frame(cbind(tsne3[y1,], y[y1]))
tsne3_y1 <- tsne3_y1[sample(nrow(tsne3_y1), 800),]
colnames(tsne3_y0) <- c("x", "y", "Not dropout")
colnames(tsne3_y1) <- c("x", "y", "dropout")
col_ndrop <- rgb(0.988, 0.459, 0.439)
col_drop <- rgb(0, 0.757, 0.769)
points3d(tsne3_y0, col=col_ndrop, size=5)
points3d(tsne3_y1, col=col_drop, size=5)
axes3d()
legend3d("topright", c("not dropout", "dropout"),  pch = 16, col=c(col_ndrop,col_drop),cex=1, inset=c(0.02))
writeWebGL(dir="../figure/tsne3_3",width=1000, height=1000)
