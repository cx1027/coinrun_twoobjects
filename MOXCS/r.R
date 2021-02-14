#library(gcookbook)
library(ggplot2)
setwd("/Users/X/PycharmProjects/multiobj")
data <- read.csv("stateWithAction_[1, 0].csv", stringsAsFactors=FALSE)
data$actionList[data$actionList==1]<-'stay'
data$actionList[data$actionList==0]<-'left'
data$actionList[data$actionList==2]<-'right'
ggplot(data, aes(x=x, y=y, colour=actionList)) + geom_point()#以颜色区分


