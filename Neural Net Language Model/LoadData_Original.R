# Load Data Original, with minor changes to accomodate R

library(R.matlab)
# library(Matrix)
#library(parallel); options(mc.cores = 4)

load_data_original <- function(N) {
     # % This method loads the training, validation and test set.
     # % It also divides the training set into mini-batches.
     # % Inputs:
     # %   N: Mini-batch size.
     # % Outputs:
     # %   train_input: An array of size D X N X M, where
     # %                 D: number of input dimensions (in this case, 3).
     # %                 N: size of each mini-batch (in this case, 100).
     # %                 M: number of minibatches.
     # %   train_target: An array of size 1 X N X M.
     # %   valid_input: An array of size D X number of points in the validation set.
     # %   test: An array of size D X number of points in the test set.
     # %   vocab: Vocabulary containing index to word mapping.
     
     data.mat <- readMat("Neural Net Language Model/data.mat")
     data <- list(testData = (data.mat$data[1,1,1][[1]]),
                  trainData = (data.mat$data[2,1,1][[1]]),
                  validData = (data.mat$data[3,1,1][[1]]),
                  vocab = unlist(data.mat$data[4,1,1])
     )
     
     numdims = size(data$trainData, 1)
     D = numdims - 1  # subtract 1 because 1:D is the number of input words and D is the predicted word
     M = floor(size(data$trainData, 2) / N)
     
     # shift to an list of M minibatches, each with D*N
     # looks like we threw out the remainder training data
     train_input <- reshape(data$trainData[1:D, 1:(N*M), drop=F], D, N, M)
     train_target <- reshape(data$trainData[D + 1, 1:(N*M), drop=F], 1, N, M)
     valid_input <- data$validData[1:D,, drop=F]
     valid_target <- data$validData[D + 1, , drop=F]
     test_input <- data$testData[1:D, , drop=F]
     test_target <- data$testData[D + 1, , drop=F]
     vocab <- data$vocab
      
     return(list(train_input=train_input,
                 train_target=train_target,
                 valid_input=valid_input,
                 valid_target=valid_target,
                 test_input=test_input,
                 test_target=test_target,
                 vocab=vocab))
}