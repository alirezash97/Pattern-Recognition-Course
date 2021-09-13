library(ramify)

init <- initial
emis <- emissions
obs <- test_letters



row <- lenght_train_letter
cols <- length(obs)
print(cols)

y = rep(0, cols)
temp_results <- matrix(0, row, cols)

for (i in 0:cols){
  for (j in 0:row){
    temp_results[j, i] = emis[j, i]
  }
}
y = argmax(temp_results, rows = F)

final_result <- c()
for (i in y){
  final_result <- paste(final_result, substring(train_letters_string, i, i))
}

print(final_result)

