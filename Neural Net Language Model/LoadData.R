# Load Data Original, with minor changes to accomodate R

library(R.matlab)
# library(Matrix)
library(parallel); options(mc.cores = 4)
library(itertools)

load_data <- function(N) {
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
     start <- seq.int(1, N*M, by=1000)
     end <- seq.int(1000, N*M, by=1000)
     train_input <- mapply(function(x, start, end) x[,start:end], 
                                           start=seq.int(1, N*M, by=1000), 
                                           end=seq.int(1000, N*M, by=1000), 
                                           MoreArgs=list(x=data$trainData[1:D, 1:(N*M)]), SIMPLIFY=F)
     train_target <- mapply(function(x, start, end) x[,start:end], 
                            start=seq.int(1, N*M, by=1000), 
                            end=seq.int(1000, N*M, by=1000), 
                            MoreArgs=list(x=data$trainData[D + 1, 1:(N*M), drop=F]), SIMPLIFY=F)
     valid_input <- data$validData[1:D,, drop=F]
     valid_target <- data$validData[D + 1, , drop=F]
     test_input <- data$validData[1:D, , drop=F]
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

# faster to access lists than indices in an array
# but slower to create a set of lists than to reshape the training data into an array



# train_input3 <- vector(M, "list")
# fn <- function() {
#      it <- isplitCols(data$trainData[1:D, 1:(N*M)], chunks = M)
#      replicate(M, nextElem(it), simplify=F)
# }
# # 
# fn2 <- function() {
#      as.list(enumerate(isplitCols(data$trainData[1:D, 1:(N*M)], chunks = M)))
# }
#      
myReshape <- function(A, ...) {
     if (!is.array(A)) {
          stop(sprintf("argument %s must be matrix or array", sQuote("A")))
     }
     nargs <- length(dots <- list(...))
     dims <- as.integer(if (nargs == 1 && matlab:::is.size_t(dots[[1]])) {
          dots[[1]]
     } else {
          unlist(dots)
     })
     if (!(length(dims) > 1)) {
          stop("dimensions must be of length greater than 1")
     }
     else if (!(all(dims > 0))) {
          stop("dimensions must be a positive quantity")
     }
     else if (prod(dims) != prod(dim(A))) {
          stop("number of elements must not change")
     }
     dim(A) <- dims
     
     return(A)
 
}

# fn3 <- function() {
#      res <- split(x=data$trainData[1:D, 1:(N*M), drop=F],
#                   f=as.factor((seq_len(D*N*M)-1) %% M))  
#      lapply(res, function(x) { dim(x) <- c(3, 1000); return(x) })
# }
# # 
# # 
# # 
# #      
# benchmark(
#      train_input1 <- reshape(data$trainData[1:D, 1:(N*M), drop=F], D, N, M),                        # 1.5x slower
#      train_input2 <- lapply(1:M, splitMatrixIntoBatch, N=N, dat=data$trainData[1:D,], byCol=TRUE), # 1.5 x slower
#      train_input3 <- fn(),  # 3.8x slower
#      train_input4 <- myReshape(data$trainData[1:D, 1:(N*M), drop=F], D, N, M), # fastest
#      train_input5 <- fn2(), # 5x slower
#      train_input6 <- mapply(function(x, start, end) x[,start:end], 
#                             start=seq.int(1, N*M, by=1000), 
#                             end=seq.int(1000, N*M, by=1000), 
#                             MoreArgs=list(x=data$trainData[1:D, 1:(N*M)]), SIMPLIFY=F), # slightly better than lapply or reshape; slower if using parallel
#      #train_input7 <- fn3(),  # 100x slower 
#      replications=100
#      )
# # 
# # 
# # 
# library(microbenchmark)
# 
# mfn1 <- function() reshape(data$trainData[1:D, 1:(N*M), drop=F], D, N, M)
# mfn2 <- function() lapply(1:M, splitMatrixIntoBatch, N=N, dat=data$trainData[1:D,], byCol=TRUE)
# mfn3 <- function()  {
#      it <- isplitCols(data$trainData[1:D, 1:(N*M)], chunks = M)
#      replicate(M, nextElem(it), simplify=F)
# }
# mfn4 <- function() myReshape(data$trainData[1:D, 1:(N*M), drop=F], D, N, M)
# mfn5 <- function() as.list(enumerate(isplitCols(data$trainData[1:D, 1:(N*M)], chunks = M)))
# mfn6 <- function() mapply(function(x, start, end) x[,start:end], 
#                           start=seq.int(1, N*M, by=1000), 
#                           end=seq.int(1000, N*M, by=1000), 
#                           MoreArgs=list(x=data$trainData[1:D, 1:(N*M)]), SIMPLIFY=F)
# 
# m <- microbenchmark(
#      mfn1(),  # second best
#      mfn2(),  # second best
#      mfn3(),  # worst
#      mfn4(),  # best
#      mfn5(),  # worst
#      mfn6(),  # second best
#      times=1000)
# 
# print(m)
# boxplot(m)

# 
# res <- split(x=data$trainData[1:D, 1:(N*M), drop=F],
#              f=as.factor((seq_len(D*N*M)-1) %% M))
# 
# fn1 <- function(x) matrix(x, nrow=3, ncol=1000)
# fn2 <- function(x) { dim(x) <- c(3, 1000); return(x) }
# 
# m <- microbenchmark(
#      lapply(res, fn1),
#      lapply(res, fn2),
#      times=1000
# )
# 
# print(m)
# boxplot(m)
# 
# 
# 
# start <- seq.int(1, N*M, by=1000)
# end <- seq.int(1000, N*M, by=1000)
# 
# res <- mapply(function(x, start, end) x[,start:end], start=start, end=end, MoreArgs=list(x=data$trainData[1:D, 1:(N*M)]), SIMPLIFY=F)

# benchmark(
#      data1 <- load_data_original(batchsize),
#      data2 <- load_data(batchsize),
#      
#      replications <- 10
#      )
# 
# benchmark(
#      for(m in 1:372) tmp1 <- data1$train_input[,,m],
#      for(m in 1:372) tmp2 <- data2$train_input[[m]],
#      replications <- 2
#           )
