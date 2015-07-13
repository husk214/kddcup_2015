library(GGally)
set.seed(777)
X <- as.matrix(read.csv("../dataset/train/10-3_1.csv", header=F ))
subsam <- sample(nrow(X), 1000)
subx <- X[subsam, ]

num_show <- 4
for (index in  165:(ceiling( (ncol(X) - 1)/num_show )-1) ) {
  show_feature <- c( (1+index*num_show):min((num_show+index*num_show), ncol(X)-1), ncol(X) )
  subx1 <- as.data.frame(subx[, show_feature])
  subx1$V717 <- as.factor(subx1$V717)

  gg = ggpairs(na.omit(subx1), lower=list(continuous="smooth"), colour="V717", params=list(corSize=4,labelSize=4));
  file_name <- paste("../figure/pair_plot2/pair_", as.character(index), ".pdf", sep="")
  pdf(file_name);
  print(gg)
  dev.off()
}
