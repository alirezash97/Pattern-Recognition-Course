library(reader)
library(ggplot2)
library(text2vec)
library(tm)
library(caret)
library(textrecipes)
library(Matrix)
library(uwot)
library(keras)
library(randomForest)


whole_dataset <- as.data.frame(read.delim("TaskCE train.txt", header=FALSE))

# pick 1200 random samples for data set
set.seed(123)
dataset_ind <- sample(seq_len(nrow(whole_dataset)), size = 3290)
dataset <- whole_dataset[dataset_ind, ]

# pick 1000 samples for train set
train_ind <- sample(seq_len(nrow(dataset)), size = 3000)
trainset <- dataset[train_ind, ]

# pick 200 samples for test set
testset <- dataset[-train_ind,]


# derive some insights
ggplot(trainset, aes(x = reorder(V2, V3), y = V3)) +
  geom_col() +
  labs(title="Tweets ",
       x = "Keywords",
       y = "Popularity") +
  coord_flip()
################


# vectorization 

num_words <- 30000
max_length <- 64
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length, 
)


#################

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

preprocess_data <- function(x){
  
  x <- tolower(x) # lower Case
  
  x <- removeWords(x, stopwords('en')) # remove stop words
  
  x <- removeNumbers(x) # remove numbers
  
  x <- removePunctuation(x) # remove punctuation
  
  x <- stripWhitespace(x) # remove white space 
  
  
  # vectorization  
  text_vectorization %>% 
    adapt(x)
  x <- text_vectorization(matrix(x, ncol = 1))
  
  # normalize data
  
  gc() # clear useless data from memory
  
  # u-map dimensionality reduction
  x <- umap(as.matrix(x), n_neighbors = 32, n_components = 256)
  
  
  return(x) }


# train inputs and labels
x_train <- preprocess_data(trainset$V4)
y_train <- factor(trainset$V3, levels=c("-2", "-1", "0", "1", "2"), ordered=TRUE)

# test inputs and labels 
x_test <- preprocess_data(testset$V4)
y_test <- factor(testset$V3, levels=c("-2", "-1", "0", "1", "2"), ordered=TRUE)



# y_train <- to_categorical(y_train, 5)
# y_test <- to_categorical(y_test, 5)

gc() # clear useless data from memory


x_train_dataframe <- as.data.frame(x_train)
y_train_dataframe <- as.data.frame(y_train)


my_trainset <- cbind(x_train_dataframe, y_train_dataframe)



# derive some insights about distributions
barplot(table(my_trainset$y_train))
# handle imbalance data
set.seed(234)
my_trainset <- upSample(x = my_trainset[, -c(y_train)],
                     y = my_trainset$y_train)

# check if it works
barplot(table(my_trainset$y_train))


# random shuffle
set.seed(42)
rows <- sample(nrow(my_trainset))
my_trainset <- my_trainset[rows, ]


# drop useless column 
drops <- "Class"
my_trainset <- my_trainset[ , !(names(my_trainset) %in% drops)]

# Set a random seed
set.seed(51)
# Training using ‘random forest’ algorithm
model <- train(y_train ~ ., # Survived is a function of the variables we decided to include
               data = my_trainset, # Use the train data frame as the training data
               method = 'rf',# Use the 'random forest' algorithm
               trControl = trainControl(method = 'cv', # Use cross-validation
                                        number = 5)) # Use 5 folds for cross-validation



# Evaluation 
x_test_dataframe <- as.data.frame(x_test)
y_test_dataframe <- as.data.frame(y_test)

my_testset <- cbind(x_test_dataframe, y_test_dataframe)

predictions <- predict(model, x_test)
confusionMatrix(data = predictions, reference = y_test)
