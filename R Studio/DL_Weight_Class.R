library(reader)
library(ggplot2)
library(text2vec)
library(tm)
library(caret)
library(textrecipes)
library(Matrix)
library(uwot)
library(keras)


whole_dataset <- as.data.frame(read.delim("TaskCE train.txt", header=FALSE))

# pick 1200 random samples for data set
set.seed(123)
dataset_ind <- sample(seq_len(nrow(whole_dataset)), size = 3290)
dataset <- whole_dataset[dataset_ind, ]

####################
dataset[, 3] <- dataset[, 3] + 2
#####################

# pick 1000 samples for train set
train_ind <- sample(seq_len(nrow(dataset)), size = 3290)
trainset <- dataset[train_ind, ]

# pick 200 samples for test set
#testset <- dataset[-train_ind,]


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
max_length <- 512
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
  x <- umap(as.matrix(x), n_neighbors = 16, n_components = 64)
  
  
  return(x) }


# train inputs and labels
x_train <- preprocess_data(trainset$V4)
y_train <- as.factor(trainset$V3)

# test inputs and labels 
#x_test <- preprocess_data(testset$V4)
#y_test <- as.factor(testset$V3)



#########################################################################
x_train_dataframe <- as.data.frame(x_train)
y_train_dataframe <- as.data.frame(y_train)


my_trainset <- cbind(x_train_dataframe, y_train_dataframe)

# derive some insights about distributions
barplot(table(my_trainset$y_train))
# handle imbalance data
set.seed(234)
#my_trainset <- upSample(x = my_trainset[, -which(names(my_trainset) == "y_train")],
#                        y = my_trainset$y_train)

# check if it works
#barplot(table(my_trainset$Class))

# random shuffle
set.seed(42)
rows <- sample(nrow(my_trainset))
my_trainset <- my_trainset[rows, ]


x_train <- my_trainset[,-which(names(my_trainset) == "y_train")]
x_train <- as.matrix(x_train)
y_train <- my_trainset$y_train
##########################################################################

y_train <- to_categorical(y_train, 5)
#y_test <- to_categorical(y_test, 5)

gc() # clear useless data from memory


# `^`(10,-19)

# define 4-fold cross validation test harness
kfold <- createFolds(my_trainset, k = 1)

# running through each fold of the cross-validation
for (fold in kfold){
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 128, input_shape = c(64)) %>% # input
    layer_activation_leaky_relu() %>%
    
    
    layer_dense(units = 256, kernel_regularizer = regularizer_l1_l2(l1 = 0, l2 = 0)) %>% 
    layer_activation_leaky_relu() %>%
    layer_dropout(rate = 0.25) %>% # dropping points at random in between layers to avoid over-fitting.
    
    #layer_dense(units = 512, kernel_regularizer = regularizer_l1_l2(l1 = 0, l2 = 0)) %>% 
    #layer_activation_leaky_relu() %>%
    #layer_dropout(rate = 0.5) %>% # dropping points at random in between layers to avoid over-fitting.
    
    #layer_dense(units = 128, kernel_regularizer = regularizer_l1_l2(l1 = 0, l2 = 0)) %>% 
    #layer_activation_leaky_relu() %>% 
    #layer_dropout(rate = 0.125) %>% # dropping points at random in between layers to avoid over-fitting.
    
    
    layer_dense(units=64) %>% 
    layer_activation_leaky_relu() %>% 
    
    layer_dense(units = 5, activation = "softmax") # output
  
  # summary of the model.
  # summary(model)
  
  # compiling the model
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr=0.005),
    metrics = c("accuracy")
  )
  
  # fitting the model
  history <- model %>% fit(
    x_train, y_train,
    class_weight = list("0"=50, "1"= 5, "2"=2, "3"=1, "4"=8),
    epochs = 6000, batch_size = 64,
    validation_split = 0.1
  )
  
  # evaluating the performance of the model
  # model %>% evaluate(x_test, y_test, verbose = 0)
  # model %>% predict_classes(x_test)
}
#####################################################

#model %>% evaluate(x_test, y_test, verbose = 0)


