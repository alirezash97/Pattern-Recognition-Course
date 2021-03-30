install.packages("keras")
library(keras)
install_keras()


library(imager)
library(nnet)
library(keras)
library(reticulate)


np <- import("numpy")



x_train <- np$load("train_images.npy")
y_train <- np$load("train_labels.npy")
x_test <- np$load("test_images.npy")
y_test <- np$load("test_labels.npy")



dim(x_train) <- c(nrow(x_train), 4096)
dim(x_test) <- c(nrow(x_test), 4096)
# rescale
x_train <- x_train / 255
x_test <- x_test / 255


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(4096)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")


summary(model)


model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)


history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


plot(history)


model %>% evaluate(x_test, y_test,verbose = 0)


random_number <- sample(0:20000, 1)
sample <- x_test[random_number, ]
suit_dim_sample <- sample
dim(suit_dim_sample) <- c(1, 4096)
predicted_value <- predict(model, suit_dim_sample)
predicted_value <- apply(predicted_value, 1, which.is.max)
sample <- sample * 255
sample <- matrix(sample, nrow = 64, byrow = TRUE)
sample <- as.cimg(sample)
plot(sample, main=sprintf("predicted value : %s", predicted_value-1))
