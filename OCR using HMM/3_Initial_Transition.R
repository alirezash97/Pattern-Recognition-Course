library(readr)


fileName <- "HMM/12345.txt"
lines <- readLines(paste(fileName, sep = ""))


temp_file = rep()
temp_word = rep()


for (line in lines) {
  
  line_words <- strsplit(line, split=" ")
  for (word in line_words){
    temp_word <- c(temp_word, word)
  }
  line_letter <- strsplit(line, split="")
  for (letter in line_letter){
    temp_file <- c(temp_file, letter)
  }
}


##############
lenght_train_letter <- nchar(train_letters)
temp_init <- rep(0, lenght_train_letter)
for (i in 1:lenght_train_letter){
  for (word in temp_word){
    if ( substring(train_letters, i, i) == substring(word, 1, 1)){temp_init[i] = temp_init[i] + 1}
  }
  
}

##############
initial <- rep()
for (i in temp_init){
  tmp <- (i+1) / (length(temp_word) + 2)
  initial <- c(initial, tmp)
}

##############
transition <-  matrix(0, lenght_train_letter, lenght_train_letter)
for (i in 1:(length(temp_file)-1)){
  if( (grepl(temp_file[i], train_letters, fixed = TRUE)) && (grepl(temp_file[i+1], train_letters, fixed = TRUE)) ){
    letter_i0_index <- unlist(gregexpr(pattern =temp_file[i], train_letters))
    letter_i1_index <- unlist(gregexpr(pattern =temp_file[i+1],train_letters))
    transition[letter_i0_index, letter_i1_index] <- transition[letter_i0_index, letter_i1_index] + 1
  }
}

##############

rows_sum <- rowSums(transition)
for (i in 1:lenght_train_letter){
  for (j in 1:lenght_train_letter){
    transition[i, j] <- (transition[i, j] + 1) / (rows_sum[i] + 2)  
  }
}

# Transition and Initial are final output in this file


