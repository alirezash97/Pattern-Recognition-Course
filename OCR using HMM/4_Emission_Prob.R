train_letters <- load_train_letter('HMM/courier-train.png')
test_letters <- load_test_letter('HMM/test-7-0.png')


noise <- 0.4
train_char_numbers <- length(train_letters)
test_char_numbers <- length(test_letters)

emissions <- matrix(0, train_char_numbers, test_char_numbers)


train_letters_string <- "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
train_letters_split <- strsplit(train_letters_string, "")[[1]]
counter <- 0
for (letter in train_letters_split){
  for (test_char_index in 1:test_char_numbers){
    
    tmp_train <- as.matrix(train_letters[letter])
    tmp_train <- tmp_train[[1]]
    if ( (dim(tmp_train)[2]) > 14 ){tmp_train <- tmp_train[, -15]}
    
    dim(tmp_train) <- c(25, 14)
    train_char_value <- tmp_train
    
    
    tmp_observation <- test_letters[test_char_index]
    tmp_observation <- tmp_observation[[1]]
    if ( (dim(tmp_observation)[2]) > 14 ){tmp_observation <- tmp_observation[, -15]}
    
    dim(tmp_observation) <- c(25, 14)
    observation <- tmp_observation
    
    missmatch <- 0
    match <- 0
    for (row in 1:25){
      for (column in 1:14){
        if (train_char_value[row, column] == observation[row, column]){
          match <- match + 1
        } else {missmatch <- missmatch + 1}
      }
    }
    emissions[(unlist(gregexpr(pattern =letter, train_letters_string))), test_char_index] <- (`^`(1-noise,match)) * (`^`(noise,missmatch))
  }
  
}








