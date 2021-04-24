library(reader)
library(caTools)
library(RWeka)
library(caret)

# load dataset
file <- "cardio_train.csv"
dataset <- read.csv(file, sep=";")
dataset$cardio = as.factor(dataset$cardio)
dataset$gender = as.factor(dataset$gender)
dataset$cholesterol = as.factor(dataset$cholesterol)
dataset$gluc = as.factor(dataset$gluc)
dataset$smoke = as.factor(dataset$smoke)
dataset$alco = as.factor(dataset$alco)
dataset$active = as.factor(dataset$active)
head(dataset)

length(dataset$cardio)


# split 
set.seed(101)
sample <- sample.split(dataset$cardio, SplitRatio = 0.7)
# trainset
train_dt <- subset(dataset, sample==T)
train_inputs <- train_dt[,-13]
train_labels <- train_dt[, 13]
#test set
test_dt <- subset(dataset, sample == F)
test_inputs <- test_dt[,-13]
test_labels <- test_dt[,13]

# train the model
cardio_ripper <- JRip(train_labels ~ ., data = train_inputs)

# show rules
cardio_ripper

# model summary
summary(cardio_ripper)

# evaluation 
ripper_pred <- predict(cardio_ripper, test_inputs)

confusionMatrix(ripper_pred, test_labels)




