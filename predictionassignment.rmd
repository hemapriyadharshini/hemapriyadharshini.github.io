---
title: "Practical Machine Learning- Predicting weight-lifting styles"
author: "Hemapriyadharshini Sampath Kumar"
date: "Sunday, February 22, 2015"
output: html_document
---

Summary:
A large amount of personal exercise activty can be obtained from devices such as Jawbone Up, Nike FuelBand, and Fitbit. Though the people quantify the level of activity they do, they rarely measure how well they do it. Thus, in this project, we make use of exercise data from accelerometers on the arm, forearm, belt, and dumbell of six participants. They were asked to perform the bareball lifts correctly and incorrectly in 5 different ways. The data is derived from the source: http://groupware.les.inf.puc-rio.br/har

6 healthy participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in the following patterns:

Exactly according to the instructions (Class A) - correct exercise pattern
Throwing the elbows to the front (Class B) - Incorrect exercise pattern
Lifting the dumbbell only halfway (Class C) - Incorrect exercise pattern
Lowering the dumbbell only halfway (Class D) - Incorrect exercise pattern and
Throwing the hips to the front (Class E) - Incorrect exercise pattern

Predict their exercise manner
Build a prediction model
Calculate the out of sample error and
Use the prediction model to predict 20 different test cases provided.

Step-1: Data preparation

In this step, the data is cleaned and processed by 
Loading the libraries
Getting the data
Loading the training and test data
Processing the data

Loading the libraries:
```{r}
library(knitr)
options(width=120)
library(caret)
library(randomForest)
library(pander)
```

Getting the training and testing data:
```{r}
downloadDataset <- function(URL="", destFile="data.csv"){
  if(!file.exists(destFile)){
    download.file(URL, destFile)
  }else{
    message("Dataset Exists.")
  }
}
```
```{r}
 trainURL<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
 downloadDataset(trainURL, "training.csv")
```




```{r}
 testURL <-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
 downloadDataset(testURL, "testing.csv")
```


Loading the data:

```{r}
training <- read.csv("C:/Users/Hemapriyadharshini/Documents/PML/training.csv",na.strings=c("NA",""))
testing <-read.csv("C:/Users/Hemapriyadharshini/Documents/PML/testing.csv",na.strings=c("NA",""))
dim(training)
```

```{r}
## [1] 19622   160
 dim(testing)
## [1]  20 160
```

Step-2: Processing the data
In this step, we find and remove the NA values in the training and the testing set.
```{r}
 sum(is.na(training))
## [1] 1921600
 sum(is.na(testing))
## [1] 2000
```

```{r}
 columnNACounts <- colSums(is.na(training)) 
 badColumns <- columnNACounts >= 19000        
 cleanTrainingdata <- training[!badColumns]        
 sum(is.na(cleanTrainingdata))  
## [1] 0
 cleanTrainingdata <- cleanTrainingdata[, c(7:60)] 
 columnNACounts <- colSums(is.na(testing))        
 badColumns <- columnNACounts >= 20                
 cleanTestingdata <- testing[!badColumns]       
 sum(is.na(cleanTestingdata)) 
## [1] 0
```

Step-3: Data Analysis

In this step, the data is analysed using summary statistics and frequency plot of the "classe" variable.
```{r}
 cleanTestingdata <- cleanTestingdata[, c(7:60)]
 e <- summary(cleanTrainingdata$classe)
 pandoc.table(e, style = "grid", justify = 'left', caption = '`classe` frequencies')

```

```{r}
## 
## 
## +------+------+------+------+------+
## | 5580 | 3797 | 3422 | 3216 | 3607 |
## +------+------+------+------+------+
## 
## Table: `classe` frequencies
```

```{r}
 plot(cleanTrainingdata$classe,col=rainbow(5),main = "`classe` frequency plot")
```
```{r fig.height=10, fig.width=10, echo=FALSE}
##plot(cleanTrainingdata$classe,col=rainbow(5),main)
```

Step-4: Data Partitioning
In this step, the cleantrainingdataset is divided into training set and testing set. Later, the model is built by executing randomForest predition algorithm
```{r}
 partition <- createDataPartition(y = cleanTrainingdata$classe, p = 0.6, list = FALSE)
 trainingdata <- cleanTrainingdata[partition, ]
testdata <- cleanTrainingdata[-partition, ]
 cvCtrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE, verboseIter = TRUE)
 model <- train(classe ~ ., data = trainingdata, method = "rf", trControl = cvCtrl)
##+ Fold1: mtry= 2 
##- Fold1: mtry= 2 
##+ Fold1: mtry=27 
##- Fold1: mtry=27 
##+ Fold1: mtry=53 
##- Fold1: mtry=53 
##+ Fold2: mtry= 2 
##- Fold2: mtry= 2 
##+ Fold2: mtry=27 
##- Fold2: mtry=27 
##+ Fold2: mtry=53 
##- Fold2: mtry=53 
##+ Fold3: mtry= 2 
##- Fold3: mtry= 2 
##+ Fold3: mtry=27 
##- Fold3: mtry=27 
##+ Fold3: mtry=53 
##- Fold3: mtry=53 
##+ Fold4: mtry= 2 
##- Fold4: mtry= 2 
##+ Fold4: mtry=27 
##- Fold4: mtry=27 
##+ Fold4: mtry=53 
##- Fold4: mtry=53 
##+ Fold5: mtry= 2 
##- Fold5: mtry= 2 
##+ Fold5: mtry=27 
##- Fold5: mtry=27 
##+ Fold5: mtry=53 
##- Fold5: mtry=53 
##Aggregating results
##Selecting tuning parameters
##Fitting mtry = 27 on full training set
```
Step-5: Prediction
In thi step, the prediction model and out of sample accuracy are calculated
```{r}
 training_pred <- predict(model, trainingdata)
 confusionMatrix(training_pred, trainingdata$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000

testing_pred <- predict(model, testdata)
confusionMatrix(testing_pred, testdata$classe)

## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    4    0    0    0
##          B    0 1513    4    0    0
##          C    0    0 1364    3    0
##          D    0    1    0 1282    1
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.997    0.997    0.999
## Specificity             0.999    0.999    1.000    1.000    1.000
## Pos Pred Value          0.998    0.997    0.998    0.998    0.999
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.998    0.998    0.998    1.000

```


Prediction assignment results against 20 testcases:

```{r}
 answer <- predict(model, cleanTestingdata)
 answer <- as.character(answer)
 answer
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"

```


Conclusion:

From the above statistics, it is evident that the prediction accuracy is 99.68% using radomForest machine learning technique.
