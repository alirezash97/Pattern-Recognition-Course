install.packages("tidyverse")
install.packages("tidytext")
install.packages("textclean")
install.packages("hunspell")
install.packages("yardstick")
install.packages("furrr")



# Data Wrangling
library(tidyverse)

# Text Preprocessing
library(tidytext)
library(textclean)
library(hunspell)

# Model Evaluation
library(yardstick)

# Deep Learning
library(keras)
use_condaenv("r-tensorflow")

library(caret)

# ggplot2 Plot Configuration
theme_set(theme_minimal() +
            theme(legend.position = "top")
)

##########################

dataset <- as.data.frame(read.delim("TaskCE train.txt", header=FALSE))
glimpse(dataset)

################################

cleansing_text <- function(x) x %>% 
  replace_non_ascii() %>% 
  tolower() %>% 
  str_replace_all(pattern = "\\@.*? |\\@.*?[:punct:]", replacement = " ") %>% 
  str_remove(pattern = "early access review") %>%
  replace_url() %>% 
  replace_hash() %>% 
  replace_html() %>% 
  replace_contraction() %>% 
  replace_word_elongation() %>% 
  str_replace_all("\\?", " questionmark") %>% 
  str_replace_all("\\!", " exclamationmark") %>% 
  str_replace_all("[:punct:]", " ") %>% 
  str_replace_all("[:digit:]", " ") %>% 
  str_trim() %>% 
  str_squish()

cleansing_text("dear @Microsoft the newOoffice for Mac is great and all, but no Lync update? C'mon!!")

###################################

library(furrr) 
plan(multisession, workers = 4) # Using 4 CPU cores

df_clean <- dataset %>% 
  mutate(
    text_clean = V4 %>% 
      future_map_chr(cleansing_text)
  ) 

######################################


word_count <- map_dbl(df_clean$text_clean, function(x) str_split(x, " ") %>% 
                        unlist() %>% 
                        length()
)

summary(word_count)

#####################################

df_clean <- df_clean %>% 
  filter(word_count > 3)

glimpse(df_clean)

######################################

df_clean[, 3] <- df_clean[, 3] + 2

######################################

set.seed(123)
row_data <- nrow(df_clean)
index <- sample(row_data, row_data*0.95)

data_train <- df_clean[ index, ]
data_test <- df_clean[-index, ]

#####################################

table(data_train$V3) %>% 
  prop.table()

data_train$V3 <- factor(data_train$V3)
# imbalance handle 
#data_train<-upSample(x=data_train$text_clean,
#                     y=data_train$V3)


####################################

# random shuffle
set.seed(42)
rows <- sample(nrow(data_train))
data_train <- data_train[rows, ]


###################################

paste(data_train$text_clean, collapse = " ") %>% 
  str_split(" ") %>% 
  unlist() %>% 
  n_distinct()

#####################################

num_words <- 8748

tokenizer <- text_tokenizer(num_words = num_words) %>% 
  fit_text_tokenizer(data_train$text_clean)

# Maximum Length of Word to use
maxlen <- 64

#####################################

train_x <- texts_to_sequences(tokenizer, data_train$text_clean) %>% 
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")

test_x <- texts_to_sequences(tokenizer, data_test$text_clean) %>% 
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")

unique(data_train[c("V3")])

# Transform the target variable on data train
train_y <- to_categorical(factor(data_train$V3), num_classes= 5)

dim(train_x)
#######################################

# Set Random Seed for Initial Weight
tensorflow::tf$random$set_seed(123)

# Build model architecture
model <- keras_model_sequential(name = "lstm_model") %>% 
  layer_embedding(name = "input",
                  input_dim = num_words,
                  input_length = maxlen,
                  output_dim = 8
  ) %>% 
  layer_lstm(name = "LSTM0",
             units = 16,
             kernel_regularizer = regularizer_l1_l2(l1 = 0.0325, l2 = 0.0325), #l1 = 0.000125, l2 = 0.000125
             return_sequences = F
  ) %>%
  layer_dropout(rate = 0.5) %>%
  #layer_lstm(name = "LSTM2",   
  #           units = 128,
  #           kernel_regularizer = regularizer_l1_l2(l1 = 0.00086, l2 = 0.00086), #l1 = 0.00025, l2 = 0.00025
  #           return_sequences = T
  #) %>%
  #layer_dropout(rate = 0.5) %>%
  #layer_lstm(name = "LSTM3",
  #           units = 32,
  #           kernel_regularizer = regularizer_l1_l2(l1 = 0, l2 = 0), #l1 = 0.000125, l2 = 0.000125
  #           return_sequences = F
  #) %>% 
  layer_dense(name = "Output",
              units = 5,
              activation = "softmax"
  )

model

#########################################

model %>% 
  compile(optimizer = optimizer_adam(lr = 0.0005),
          metrics = "accuracy",
          loss = "categorical_crossentropy"
  )

epochs <- 80
batch_size <- 32



train_history <- model %>% 
  fit(x = train_x,
      y = train_y,
      batch_size = batch_size,
      class_weight = list("0"=50, "1"= 5, "2"=2, "3"=1, "4"=8),
      epochs = epochs,
      validation_split = 0.1, # 10% validation data
      
      # print progress but don't create graphic
      verbose = 1,
      view_metrics = 0
  )

plot(train_history) +
  geom_line()

##############################
table(data_test$V3) %>% 
  prop.table()


pred_class <- factor(predict_classes(model, test_x))

head(pred_class, 10)

true_class <- factor(data_test$V3)
head(true_class, 10)

confusionMatrix(data = pred_class, 
                reference = true_class) 

