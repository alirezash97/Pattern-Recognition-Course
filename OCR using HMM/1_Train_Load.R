library(imager)


char_width <- 14
char_height <- 25
train_letters <- "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

load_train_letter <- function(img_name){
  
  img <- load.image(img_name)
  img_width <-  width(img)
  img_height <- height(img)
  
  len_result <- floor(img_width / char_width)
  result <- vector(mode="list", length=len_result)
  names(result) <- c("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
                     "Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j",
                     "k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","0","1","2","3",
                     "4","5","6","7","8","9")
  
  img <- t(as.matrix(img))
  j = 1
  for (i in seq(0, img_width-14, by=char_width)){
    result[j] <- list(img[,i:(i+14)])
    j = j+1
  }
  
  return(result)
  
}

#result <- load_train_letter('HMM/courier-train.png')
#result[69]
