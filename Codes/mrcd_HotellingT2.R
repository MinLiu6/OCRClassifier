if(!require('rrcov')) {
  install.packages('rrcov')
  library('rrcov')
}
library(Matrix)
library(matlib)
library(ggplot2)
library(ggpubr)
library(RColorBrewer)



# depth <- read.table('F:/AlmWork/Science/OCR/src/CL_data/HK_CCR/HK_CCRdepth.txt',header=FALSE,sep='\t')
# adjustdWps <- read.table('F:/AlmWork/Science/OCR/src/CL_data/HK_CCR/HK_CCRadjustWps.txt',header=FALSE,sep='\t')
# Uend <- read.table('F:/AlmWork/Science/OCR/src/CL_data/HK_CCR/HK_CCRsmoothUend.txt',header=FALSE,sep='\t')
# Dend <-read.table('F:/AlmWork/Science/OCR/src/CL_data/HK_CCR/HK_CCRsmoothDend.txt',header=FALSE,sep='\t')
# depth <- read.table('F:/AlmWork/Science/OCR/src/CL_data/HK/HK_depth.txt',header=FALSE,sep='\t')
# adjustdWps <- read.table('F:/AlmWork/Science/OCR/src/CL_data/HK/HK_adjustWps.txt',header=FALSE,sep='\t')
# Uend <- read.table('F:/AlmWork/Science/OCR/src/CL_data/HK/HK_Uend.txt',header=FALSE,sep='\t')
# Dend <-read.table('F:/AlmWork/Science/OCR/src/CL_data/HK/HK_Dend.txt',header=FALSE,sep='\t') 

depth <- read.table('F:/AlmWork/Science/OCR/src/mrcdHotelling/depth_20k/1depth_59760000.txt',header=FALSE,sep='\t')
adjustdWps <- read.table('F:/AlmWork/Science/OCR/src/mrcdHotelling/adjusted_20k/1adjustedWps_59760000.txt',header=FALSE,sep='\t')
Uend <- read.table('F:/AlmWork/Science/OCR/src/mrcdHotelling/Uend_20k/1Uend_59760000.txt',header=FALSE,sep='\t')
Dend <-read.table('F:/AlmWork/Science/OCR/src/mrcdHotelling/Dend_20k/1Dend_59760000.txt',header=FALSE,sep='\t')


dataWinsum <- function(data,winSize){
  data <- as.array(data)
  n <- length(data)
  print(n)
  dataWinSumList <- rep(0,2)
  count <- n%/%winSize
  #print(count)
  for(i in 1:count){
    start <- winSize*(i-1)+1
    end <- winSize*i
    print(c(start,end))
    dataWinSumList[i] <- sum(data[start:end])/winSize
  }
  return (dataWinSumList)
}

N <- nrow(depth)
flag <- 1
T2Max <- rep(0,2)
depthWin<- rep(0,2)
adjustdWpsWin<- rep(0,2)
Uendwin<- rep(0,2)
DendWin<- rep(0,2)
depthWinList <- rep(0,2)


#取第三列值
x1 <- depth[,3]
x2 <- adjustdWps[,3]
x3 <- Uend[,3]
x4 <- Dend[,3]
#计算窗口内的均值
winSize <- 200
depthWin <- dataWinsum(x1,winSize)
adjustdWpsWin <-dataWinsum(x2,winSize)
Uendwin <-dataWinsum(x3,winSize)
DendWin <-dataWinsum(x4,winSize)

#构造100行4列矩阵
dataMatrix<- cbind(depthWin,adjustdWpsWin,Uendwin,DendWin)
#计算MRCD估计的HotellingT2
out <- CovMrcd(dataMatrix, alpha=0.5)

center <- mean(dataMatrix)
covM <- cov(dataMatrix)
icovM <- solve(covM)
n <- nrow(dataMatrix)
p <- ncol(dataMatrix)
T2List <- rep(0,2)
for(j in 1:n){
  T2List[j] <- t(as.matrix(dataMatrix[j,]-out$center))%*%((out$icov)%*%(as.matrix(dataMatrix[j,]-out$center)))
  #T2List[j] <- t(as.matrix(dataMatrix[j,]-center))%*%((icovM)%*%(as.matrix(dataMatrix[j,]-center)))
}
contig <-depth[1,1] 
alpha <- 0.05
m <- mean(T2List)
v <- var(T2List)
q <- 1/(v*p/(m*2)-1)*(p+2)+4
d <- m*(q-2)/q
det_Value <- det(out$cov)

#ucl=((n-1)**2/n)*pbeta(1-alpha, p / 2, (n - p - 1) / 2)
ucl <- d*qf(1-alpha,p,q)
start <- depth[1,2]
ndrstart <- 0
ndrend <- 0
overUcl <- vector()
#run rules

m <- 0
for(j in 1:length(T2List)){
  if(T2List[j]>ucl){
    m <- m+1
    overUcl[m] <- j
    #首点
    if(j==1){
      ndrstart <- start-winSize
      ndrend <- start+winSize*2
      x <- paste(contig,ndrstart,ndrend)
      print(x)
      #write.table(x,"F:/AlmWork/Science/RProject/MRCD/chr1_Overuclfinal.txt",append=T,quote=FALSE,row.names = FALSE,col.names = FALSE)
    }
    #尾点
    if(j==length(T2List)){
      ndrstart <- start+winSize*99
      ndrend <- start+winSize*101
      x <- paste(contig,ndrstart,ndrend)
      print(x)
      #write.table(x,"F:/AlmWork/Science/RProject/MRCD/chr1_Overuclfinal.txt",append=T,quote=FALSE,row.names = FALSE,col.names = FALSE)
    }
  }
}
k <- 1
while(k<length(overUcl)){
  cnt <- 1
  while(k<length(overUcl) & (overUcl[k+1]-overUcl[k]<=4)){
    cnt <- cnt+1
    k <- k+1
  }
  if(overUcl[k]-overUcl[k-cnt+1]>=2){
    ndrstart <- start+winSize*(overUcl[k-cnt+1]-1)
    ndrend <- start+winSize*overUcl[k]
    x <- paste(contig,ndrstart,ndrend)
    print(x)
    #write.table(x,"F:/AlmWork/Science/RProject/MRCD/chr1_Overuclfinal.txt",append=T,quote=FALSE,row.names = FALSE,col.names = FALSE)
  }
  
  else{
    k <- k+1
  }
}

library(mvnormtest)
#data <- data.frame(dataMatrix)
data=matrix()
j=0
for(k in 1:length(T2List)){
  if(T2List[k]<ucl){
    j=j+1
    print(j)
    print(dataMatrix[k,])
    data[j] <- dataMatrix[k,]
  }
}
data <- t(data)
mshapiro.test(data)

library(jpeg)
jpeg(filename = "F:/AlmWork/Science/RProject/MRCD/figure3a.jpg",
     width = 18/2.54, height = 16/2.54, units = "in",quality = 90,res = 900)
plot(c(1:length(T2List)*200+start),T2List,pch=15,col='blue',ylab='T square statistics',xlab='Reference genome coordinates(bp)',family = "serif")
abline(h=ucl,lty=3,col = "red",lwd=3)
text(6*200+start,ucl+2,'UCL',col = "red")
lines(c(1:length(T2List)*200+start),T2List,col="blue",lty=1)
dev.off()

# plot(c(1:length(T2List)*200+start),T2List,pch=15,cex.axis=2,cex.lab=2,cex.main=2,col='blue',ylab='T square statistics',xlab='参考基因组坐标(bp)',family = "serif", fontface = "bold")
# abline(h=ucl,lty=3,col = "red")
# text(6*200+start,ucl+3,'UCL',col = "red",cex=2)
# lines(c(1:length(T2List)*200+start),T2List,col="blue",lty=1)

#write.table(T2List,file = "F:/AlmWork/Science/RProject/MRCD/3284.txt", sep ="\n", row.names =FALSE, col.names =FALSE, quote =FALSE)

#写入文件contig mean(T2) var(T2) d q det(cov)

#x <- paste(contig,round(m,4),round(v,4),round(d,4),round(q,4),det_Value)
#write.table(x,"F:/AlmWork/Science/RProject/MRCD/ATAC_CCR.txt", append=T,quote=FALSE,row.names = FALSE,col.names = FALSE)
 


# plot(c(1:length(T2Max[1:1000,])),T2Max[1:1000,],type="o")
# abline(h=ucl,lty=3,col = "darkgreen")
# #画箱型图
# T2Max <- cbind(T2Max)
# T2Max <- as.data.frame(T2Max)
# #T2Max$T2Max <- as.factor(T2Max$T2Max)
# #p <- ggplot(T2Max)+geom_boxplot()
# p <- ggboxplot(T2Max,color = "supp", palette = "jama",add = "jitter")+stat_compare_means()
# p