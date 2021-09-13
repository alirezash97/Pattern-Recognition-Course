load_test_letter <- function(img_name){
  
  img <- load.image(img_name)
  img_width <-  width(img)
  img_height <- height(img)
  
  len_result <- floor(img_width / char_width)
  result <- vector(mode="list", length=len_result)
  
  img <- t(as.matrix(img))
  j = 1
  for (i in seq(0, img_width-14, by=char_width)){
    result[j] <- list(img[,i:(i+14)])
    j = j+1
  }
  
  return(result)
  
}

#result <- load_test_letter('HMM/test-1-0.png')
#result
