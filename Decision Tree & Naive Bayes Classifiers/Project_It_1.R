#Bhumit Shah 1001765834
#Kaustubh Rajpathak 1001770219
#Project 1


setwd("C:/Users/kkr0219/Documents/Data Mining Datasets/bank-additional-full")
bankData <- read.csv(file = 'bank-additional-full.csv',header=TRUE, sep=";")

#Cleaning and pre-processing

#Removing rows with unknown values
nrow(bankData[bankData$job != "unknown" & bankData$education != "unknown" & bankData$marital != "unknown" & bankData$default != "unknown" 
              & bankData$housing != "unknown" & bankData$loan != "unknown" & bankData$contact != "unknown", ])
tempdata <- bankData[bankData$job != "unknown" & bankData$education != "unknown" & bankData$marital != "unknown" & bankData$default != "unknown" 
                     & bankData$housing != "unknown" & bankData$loan != "unknown" & bankData$contact != "unknown", ]
nrow(tempdata)

#Removing columns [marital,default, housing, loan, contact]
tempdata$marital <- NULL
tempdata$default <- NULL
tempdata$housing <- NULL
tempdata$loan <- NULL
tempdata$contact <- NULL

#Testing using attribute deletion
#tempdata$campaign <- NULL
#tempdata$duration <- NULL
#tempdata$cons.price.idx <- NULL

#writing final cleaned and pre processed dataset to new file
write.csv(tempdata, "C:/Users/kkr0219/Documents/Data Mining Datasets/bank-additional-full/finalpreprocessed.csv", row.names = FALSE)

preprocessed <- read.csv(file = 'C:/Users/kkr0219/Documents/Data Mining Datasets/bank-additional-full/finalpreprocessed.csv')
View(preprocessed)

#getting sample from preprocessed data with seed value 10
set.seed(10)
sampleset <- preprocessed[sample(nrow(preprocessed), 10000),]
View(sampleset)

#using 80/20 split for training and testing
sample_size <- floor(0.8 * nrow(sampleset))

set.seed(10)
train_ind <- sample(nrow(sampleset), sample_size)

trainset <- sampleset[train_ind, ]
testset <- sampleset[-train_ind, ]

View(trainset)
View(testset)

#using rpart library to implement decision tree algorithm
library(rpart)
library(rattle) #used for printing decision trees

#generation complete decision trees by setting cp = 0
#xval is the number of cross validations
ginimodelfull <- rpart(y~., data = trainset, method = 'class', control = rpart.control(cp = 0, xval = 10))
infomodelfull <- rpart(y~., data = trainset, method = 'class', parms = list(split = 'information'), control = rpart.control(cp = 0, xval = 10))

#display complete decision trees
#fancyRpartPlot(infomodelfull, palettes = c("Greens", "Reds"), sub = "")
#fancyRpartPlot(ginimodelfull, palettes = c("Greens", "Reds"), sub = "")

#print cp information to identify optimal cp for minimal xerror
printcp(infomodelfull)
printcp(ginimodelfull)

#graph for cp values vs xerror
plotcp(infomodelfull, lty = 3, col = 2, upper = "splits")
plotcp(ginimodelfull, lty = 3, col = 2, upper = "splits")

#retrieve best cp value for pruning complete decision tree
ginibestcp <- ginimodelfull$cptable[which.min(ginimodelfull$cptable[,"xerror"]),"CP"]
infobestcp <- infomodelfull$cptable[which.min(infomodelfull$cptable[,"xerror"]),"CP"]

#pruning complete decision tree
ginimodelpruned <- rpart(y~., data = trainset, method = 'class', control = rpart.control(cp = ginibestcp, xval = 10))
infomodelpruned <- rpart(y~., data = trainset, method = 'class', parms = list(split = 'information'), control = rpart.control(cp = infobestcp, xval = 10))

#display pruned decision tree
fancyRpartPlot(infomodelpruned, palettes = c("Greens", "Reds"), sub = "")
fancyRpartPlot(ginimodelpruned, palettes = c("Greens", "Reds"), sub = "")

#print attribute importance
ginivarimp <- as.data.frame(ginimodelpruned$variable.importance)
infovarimp <- as.data.frame(infomodelpruned$variable.importance)
print(paste('Variable importance according to gini index metric'))
print(ginivarimp)
print(paste('Variable importance according to information gain metric'))
print(infovarimp)

#prediction on the basis of pruned decision tree
ginimodelpredict <- predict(ginimodelpruned, testset, type = "class")
infomodelpredict <- predict(infomodelpruned, testset, type = "class")


#using e1071 library to implement naives bayes
library(e1071)
library(caret)

#training the model
#laplace smoothing applied for improving f1 score
banknb <- naiveBayes(as.factor(y)~., trainset, laplace = 4)
banknb

#predicting on the model
banknbpredict <- predict(banknb, testset, prob = TRUE)

#retrieve confusion matrix for predicted values
conf_matrix_info <- table(testset$y, infomodelpredict)
conf_matrix_gini <- table(testset$y, ginimodelpredict)
conf_matrix_nb <- table(testset$y, banknbpredict)

rownames(conf_matrix_gini) <- paste("Actual", rownames(conf_matrix_gini), sep = ":")
rownames(conf_matrix_info) <- paste("Actual", rownames(conf_matrix_info), sep = ":")
rownames(conf_matrix_nb) <- paste("Actual", rownames(conf_matrix_nb), sep = ":")
colnames(conf_matrix_gini) <- paste("Predicted", colnames(conf_matrix_gini), sep = ":")
colnames(conf_matrix_info) <- paste("Predicted", colnames(conf_matrix_info), sep = ":")
colnames(conf_matrix_nb) <- paste("Predicted", colnames(conf_matrix_nb), sep = ":")

print(conf_matrix_info)
print(conf_matrix_gini)
print(conf_matrix_nb)

#calculating various model statistics based on the confusion matrix
giniaccuracy <- sum(diag(conf_matrix_gini)) / sum(conf_matrix_gini)
giniprecision <- conf_matrix_gini[2, "Predicted:yes"] / sum(conf_matrix_gini[1, "Predicted:yes"], conf_matrix_gini[2, "Predicted:yes"])
ginirecall <- conf_matrix_gini[2, "Predicted:yes"] / sum(conf_matrix_gini[2, "Predicted:yes"], conf_matrix_gini[2, "Predicted:no"])
ginif1score <- (2*(ginirecall * giniprecision)) / (ginirecall + giniprecision)

infoaccuracy <- sum(diag(conf_matrix_info)) / sum(conf_matrix_info)
infoprecision <- conf_matrix_info[2, "Predicted:yes"] / sum(conf_matrix_info[1, "Predicted:yes"], conf_matrix_info[2, "Predicted:yes"])
inforecall <- conf_matrix_info[2, "Predicted:yes"] / sum(conf_matrix_info[2, "Predicted:yes"], conf_matrix_info[2, "Predicted:no"])
infof1score <- (2*(inforecall * infoprecision)) / (inforecall + infoprecision)

nbaccuracy <- sum(diag(conf_matrix_nb)) / sum(conf_matrix_nb)
nbprecision <- conf_matrix_nb[2, "Predicted:yes"] / sum(conf_matrix_nb[1, "Predicted:yes"], conf_matrix_nb[2, "Predicted:yes"])
nbrecall <- conf_matrix_nb[2, "Predicted:yes"] / sum(conf_matrix_nb[2, "Predicted:yes"], conf_matrix_nb[2, "Predicted:no"])
nbf1score <- (2*(nbrecall * nbprecision)) / (nbrecall + nbprecision)

print(paste('Accuracy for gini metric test is', giniaccuracy*100, '%'))
print(paste('Precision for gini metric test is', giniprecision))
print(paste('Recall for gini metric test is', ginirecall))
print(paste('F1 score for gini metric test is', ginif1score))

print(paste('Accuracy for information metric test is', infoaccuracy*100, '%'))
print(paste('Precision for information metric test is', infoprecision))
print(paste('Recall for information metric test is', inforecall))
print(paste('F1 score for information metric test is', infof1score))

print(paste('Accuracy for naives bayes model is', nbaccuracy*100, '%'))
print(paste('Precision for naives bayes model is', nbprecision))
print(paste('Recall for naives bayes model is', nbrecall))
print(paste('F1 score for naives bayes model is', nbf1score))


































