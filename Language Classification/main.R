library(reader)
library(text2vec)
library(tm)
library(caret)
library(textrecipes)
library(sGPCA)
library(qlcMatrix)
library(uwot)




dataset <- read.csv("dataMining.csv")

dataset <- dataset[-c(107), ] # wrong class

clean <- function(x){
  
  x <-tolower(x)
  
  x <- removeWords(x, stopwords('en'))
  
  x <- removeWords(x, stopwords('fr'))
  
  x <- removeWords(x, stopwords('de'))
  
  x <- removeWords(x, stopwords('spanish'))
  
  x <- removeNumbers(x)
  
  x <- removePunctuation(x)
  
  x <- stripWhitespace(x)
  
  tokens <- word_tokenizer(x)
  dtm <- create_dtm(itoken(tokens), hash_vectorizer())
  model_tfidf <- TfIdf$new()
  x <- model_tfidf$fit_transform(dtm)
  
  return(x) }

inputs <- clean(dataset$content)
inputs <- umap(as.matrix(inputs),n_neighbors = 10, n_components = 128)

labels <-as.factor(dataset$class)






## 85% of the sample size
smp_size <- floor(0.85 * nrow(inputs))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(inputs)), size = smp_size)


x_train <- inputs[train_ind, ]
y_train <- labels[train_ind]
x_test <- inputs[-train_ind, ]
y_test <- labels[-train_ind]


x_train <- as.data.frame(as.matrix(x_train))
x_test <- as.data.frame(as.matrix(x_test))


ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 15) # 10-fold CV


set.seed(100)  

rf.tfidf <- train(x_train, y_train,
                  method = "cforest", trControl = ctrl) # train random forest

rf.tfidf

predictions <- predict(rf.tfidf, x_test)
confusionMatrix(data = predictions, reference = y_test)

