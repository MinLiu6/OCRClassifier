library(ICSNP)
library(MASS)
library(ggplot2)
library(HypergeoMat)
library(mvtnorm)
library(randomForest)
library(varSelRF)
library(pROC)
library(mlr)
library(rrcov)

ocrfedata2 <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/27filter_ocr_feature_x.txt',header=FALSE,sep='\t')
ccrfedata2 <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/27filter_ccr_feature_x.txt',header=FALSE,sep='\t')
mixfedata2 <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/27filter_0.8ocr_feature_x.txt',header=FALSE,sep='\t')

pcdata <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/filter_rate0.8_pc.txt',header=FALSE)
ocrWindata2 <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/27filter_ocr_winsum_x.txt',header=FALSE,sep='\t')
ccrWindata2 <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/27filter_ccr_winsum_x.txt',header=FALSE,sep='\t')
mixWindata2 <- read.table('F:/AlmWork/Science/OCRFinder-main/content2/recentData/27filter_0.8ocr_winsum_x.txt',header=FALSE,sep='\t')

pc <- pcdata[,1]

ccrFe1 <- ccrfedata2[1:600,]
ccrFe2 <- ccrfedata2
ocrFe2 <- ocrfedata2
mixFe2 <-mixfedata2

ccrWin1 <- ccrWindata2[1:600,]
ccrWin2 <- ccrWindata2
ocrWin2 <- ocrWindata2
mixWin2 <-mixWindata2

ccrFeature1 <- ccrWin1
ccrFeature2 <- ccrWin2
ocrFeature2 <- ocrWin2
mixFeature2 <-mixWin2

# ccrFeature1 <- cbind(ccrFe1,ccrWin1)
# ccrFeature2 <- cbind(ccrFe2,ccrWin2)
# ocrFeature2 <- cbind(ocrFe2,ocrWin2)
# mixFeature2 <-cbind(mixFe2,mixWin2)


# ccrFeature1 <- ccrWindata1[,4:7]
# 
# rate=0.8
# ccrFeature2 <- ccrWindata2[,4:7]
# ocrFeature2 <- ocrWindata2[1:nrow(ccrFeature2),4:7]
# mixFeature2 <-ocrFeature2*rate+ccrFeature2*(1-rate)

# plot(ccrFeature2$V7, ccrFeature2$V8,
#      cex=1,lwd=2,ylim=c(0,1),xlim=c(-1.0,1),
#      pch=1,col='#000000',xlab='x1',ylab='x2')
# points(ocrFeature2$V7, ocrFeature2$V8,
#        pch=22,col='#F08080',cex=1,lwd=2,)
# points(mixFeature2$V7, mixFeature2$V8,
#        pch=4,col='#9ACD32',cex=1,lwd=2,)


##############################MEWMA##################
ht <- rep(0,2)
icht <- rep(0,2)
bootstap_icht <- rep(0,2)
fd <- rep(0,2)
pt <- rep(0,2)
dt <- 0
lamda <- 0.2

icdata <- as.matrix(ccrFeature1[,c(1,2)])
data1 <- as.matrix(icdata)
icm <- nrow(data1)
icp <- ncol(data1)

testData <- rbind(ccrFeature2[,c(1,2)],ocrFeature2[,c(1,2)],mixFeature2[,c(1,2)])
test <- testData
m <- nrow(test)#观测值数量
p <- ncol(test)#变量数
data2 <- as.matrix(test)
plot(ccrFeature2[,1], ccrFeature2[,2],
     cex=1,lwd=2,xlim=c(1,50),ylim=c(1,100),
     pch=1,col='#000000',xlab='depth',ylab='wps')
points(ocrFeature2[,1], ocrFeature2[,2],
       pch=22,col='#F08080',cex=1,lwd=2,)
points(mixFeature2[,1], mixFeature2[,2],
       pch=4,col='#9ACD32',cex=1,lwd=2,)

library(mvnormtest)
mshapiro.test(t(ccrFeature2[,c(1,2)]))
qqnorm(t(ccrFeature2[,c(1,2)]))

out <- CovMrcd(icdata, alpha=0.5)
center <- colMeans(icdata)#均值
# cov_mat <- cov(icdata)#协方差矩阵
cov_mat <- out$cov
icovM <- solve(cov_mat)

icZ <- matrix(nrow = icm,ncol = icp)
icZ[1,] <- 0
Z_covmat <- (lamda/(2-lamda))*cov_mat
Z_covM <- solve(Z_covmat)

Z <- matrix(nrow = m,ncol = p)
Z[1,] <- 0
for(i in 2:icm){
  icZ[i,] <- lamda*data1[i,]+(1-lamda)*icZ[i-1,]
  icht[i] <- t(as.matrix(icZ[i,]))%*%(Z_covM)%*%(as.matrix(icZ[i,]))
  
}
for(j in 2:m){
  Z[j,] <- lamda*data2[j,]+(1-lamda)*Z[j-1,]
  ht[j] <- t(as.matrix(Z[j,]))%*%(Z_covM)%*%(as.matrix(Z[j,]))
}

alpha <- 0.05
B <- 100
dtList <- rep(0,2)
for(r in 1:B){
  bootstap_icht <- sample(icht,B,replace = TRUE)#没有放回抽样
  dtList[r] <- quantile(bootstap_icht,(1-alpha))
}

dt <- mean(dtList)
fd <- ht-dt
pt <- 1/(1+exp(-fd))


# plot(c(1:length(pt)),pt,xlab="观测值",ylab="监控统计量")
# plot(c(1:length(pc)),pc,col='#F08080')

############################ST控制图的控制限####################################################
aita <- 0.3
icst <- rep(0,2)
icdelta <- rep(0,2)

icfd <- icht-dt
icpt <- 1/(1+exp(-icfd))
icpc <- pc[1:600,]

icdelta <- abs(0.5-icpc)*2
icst <- aita^icdelta*icpt+(1-aita^icdelta)*icpc

qu_bst <- rep(0,2)
for(k in 1:B){
  bootstap_st <- sample(icst,B,replace = TRUE)#有放回抽样
  qu_bst[k] <- quantile(bootstap_st,(1-alpha))
}
cl <- mean(qu_bst)

############################ST控制图####################################################
# aita <- 0.3
st <- rep(0,2)
delta <- rep(0,2)
delta <- abs(0.5-pc)*2
# st <- aita^delta*pt+(1-aita^delta)*pc

st <- aita^delta*pt+(1-aita^delta)*pc

l1 <- length(ccrWin2[,1])
l2 <- length(ocrWin2[,1])
l3 <- length(mixWin2[,1])


library(jpeg)
jpeg(filename = "F:/AlmWork/Science/RProject/MRCD/ST_Feature/st.jpg",
     width = 18/2.54, height = 12/2.54, units = "in",quality = 90,res = 900)
plot(c(1:l1),st[1:l1],pch=1,type='o',xlab="观测值",ylab="监控统计量",col='#000000',
     xlim=c(1,length(st)),ylim=c(0,1.5))
points(c((l1+1):(l1+l2)),st[(l1+1):(l1+l2)],pch=22,type='o',col='#F08080')
points(c((l1+l2+1):(l1+l2+l3)),st[(l1+l2+1):(l1+l2+l3)],pch=4,type='o',col='#9ACD32')
abline(h=cl,lty=3,col = "red")
legend(x='topright',
       legend = c('闭合状态','开放状态','部分开放状态'),
       pch=c(1,22,4),col=c('#000000','#F08080','#9ACD32'),
       pt.cex=c(1,1,1),pt.lwd = c(0.5,0.5,0.5),
       text.font=0.5)
dev.off()

jpeg(filename = "F:/AlmWork/Science/RProject/MRCD/ST_Feature/pt.jpg",
     width = 18/2.54, height = 12/2.54, units = "in",quality = 90,res = 900)
plot(c(1:l1),pt[1:l1],xlab="观测值",ylab="pt",col='#000000',
      xlim=c(1,length(st)),ylim=c(0,1.5))
points(c((l1+1):(l1+l2)),pt[(l1+1):(l1+l2)],pch=22,col='#F08080')
points(c((l1+l2+1):(l1+l2+l3)),pt[(l1+l2+1):(l1+l2+l3)],pch=4,col='#9ACD32')
legend(x='topright',
       legend = c('闭合状态','开放状态','部分开放状态'),
       pch=c(1,22,4),col=c('#000000','#F08080','#9ACD32'),
       pt.cex=c(1,1,1),pt.lwd = c(0.5,0.5,0.5),
       text.font=0.5)
abline(h=0.5,lty=3,col = "red")
dev.off()

jpeg(filename = "F:/AlmWork/Science/RProject/MRCD/ST_Feature/pc.jpg",
     width = 18/2.54, height = 12/2.54, units = "in",quality = 90,res = 900)
plot(c(1:l1),pc[1:l1],xlab="观测值",ylab="pc",col='#000000',
     xlim=c(1,length(st)),ylim=c(0,1.5))
points(c((l1+1):(l1+l2)),pc[(l1+1):(l1+l2)],pch=22,col='#F08080')
points(c((l1+l2+1):(l1+l2+l3)),pc[(l1+l2+1):(l1+l2+l3)],pch=4,col='#9ACD32')
legend(x='topright',
       legend = c('闭合状态','开放状态','部分开放状态'),
       pch=c(1,22,4),col=c('#000000','#F08080','#9ACD32'),
       pt.cex=c(1,1,1),pt.lwd = c(0.5,0.5,0.5),
       text.font=0.5)
abline(h=0.5,lty=3,col = "red")
dev.off()
#write.table(st,file ="F:/AlmWork/Science/OCRFinder-main/content2/recentData/st0.55_27train.csv", row.names = FALSE, col.names =FALSE, quote =FALSE)

