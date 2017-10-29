# Script for converting RNA-Seq .txt gene expression datasets into .csv files.
# Also standardizes values and converts categorical values.

library(tools)

files <- list.files(pattern = "\\.txt$") # Gets list of files in current directory

for(file in files){
  
  filename <- file_path_sans_ext(file) # Gets filename without extension
  
  data <- read.table(file, header = TRUE)  # Reads text file
  
  # Uncomment lines below if you want to check if read was successful
  
  # head(data)  # print first part of data
  # tail(data)  # print last part of data
  # data        # print data frame 
  

  # Encodes the class of each example as numerical values
  data$class = factor(data$class,
                      levels = c('TP', 'NT'),
                      labels = c(0, 1))

  
  # Calculates z-scores for standardization, except for
  # the last column (which indicates the class)
  data[, -ncol(data)] <- scale(data[, -ncol(data)], center = TRUE, scale = TRUE)
  
  # Creates output file
  write.csv(data, file = paste(filename, "csv", sep = "."), row.names = FALSE)
}
