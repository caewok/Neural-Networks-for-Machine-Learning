# Matlab helper functions
# Functions to replicate various matlab commands

# most are found in the matlab library
library(matlab)
library(Matrix)

# HELPER FUNCTIONS
randn <- function(x, y) matrix(data=rnorm(x*y), nrow=x, ncol=y)
myPrintf <- function(txt, ...) writeLines(sprintf(txt, ...), sep="", con=stdout(), useBytes=TRUE)
myReshape <- function(mat, nrows, ncols) {
     if(missing(ncols)) {
          dim(mat) <- c(nrows, floor(length(mat) / nrows))
     } else {
          dim(mat) <- c(floor(length(mat) / ncols), ncols)
     }
     return(mat)
}
     
myRepMat <- function(A, nrows, ncols) {
     # only works for numeric matrices with set number of rows & columns
     stopifnot(nrows > 0 & ncols > 0)
     if(is.null(dim(A))) A <- Matrix(A, ncol=1, nrow = length(A))
     
     if(missing(ncols)) ncols <- nrows
     if(nrows == 1 & ncols == 1) return(A)     
     if(ncols == 1) {
          B <- (t(myMatMult(A, create1Matrix(nrows))))
     } else if(nrows == 1) {
          B <- (myMatMult(A, create1Matrix(ncols)))  
     } else {
     #return(kronecker(matrix(1, nrow=nrows, ncol=ncols), A))
          B <- (myMatMult(A, matrix(1, nrow=nrows, ncol=ncols)))
     }
     #res <- (all.equal(B, repmat(A, nrows, ncols)))
     #if(res != TRUE) warning(res)
     
     return(B)
     
}

# tmp <- A
# benchmark(
#      dim(tmp) <- c(length(A), 1),
#      tmp <- Matrix(A, ncol=1, nrow=length(A)),
#      replications=100
#      )

myRepMat2 <- function(A, nrows, ncols) {
     # only works for numeric matrices with set number of rows & columns
     stopifnot(nrows > 0 & ncols > 0)
     if(is.null(dim(A))) A <- Matrix(A, ncol=1, nrow = length(A))
     
     if(missing(ncols)) ncols <- nrows
     if(nrows == 1 & ncols == 1) return(A)     
     if(ncols == 1) {
          B <- (t(kronecker(create1Matrix(nrows), A)))
     } else if(nrows == 1) {
          B <- (kronecker(create1Matrix(ncols), A))  
     } else {
          #return(kronecker(matrix(1, nrow=nrows, ncol=ncols), A))
          B <- (kronecker(matrix(1, nrow=nrows, ncol=ncols), A))
     }
     #res <- (all.equal(B, repmat(A, nrows, ncols)))
     #if(res != TRUE) warning(res)
     
     return(B)
}

myRepMat3 <- function(A, nrows, ncols) {
     # only works for numeric matrices with set number of rows & columns
     stopifnot(nrows > 0 & ncols > 0)
     if(is.null(dim(A))) A <- Matrix(A, ncol=1, nrow = length(A))
     
     if(missing(ncols)) ncols <- nrows
     if(nrows == 1 & ncols == 1) return(A)     
     if(ncols == 1) {
          B <- (t((A %*% create1Matrix(nrows))))
     } else if(nrows == 1) {
          B <- ((A %*% create1Matrix(ncols)))  
     } else {
          #return(kronecker(matrix(1, nrow=nrows, ncol=ncols), A))
          B <- ((A %*% matrix(1, nrow=nrows, ncol=ncols)))
     }
     #res <- (all.equal(B, repmat(A, nrows, ncols)))
     #if(res != TRUE) warning(res)
     
     return(B)
     
}

myRepMat4 <- function(A, nrows, ncols) {
     stopifnot(nrows > 0 & ncols > 0)
     
     
     if(missing(ncols)) ncols <- nrows
     if(nrows == 1 & ncols == 1) return(A)     
     if(ncols == 1) {
          B <- Matrix(A, nrow=nrow, ncol=length(A), byrow=FALSE)
               
          # B <- (t((A %*% create1Matrix(nrows))))
     } else if(nrows == 1) {
          B <- Matrix(A, nrow=length(A), ncol=ncols, byrow=TRUE)
          # B <- ((A %*% create1Matrix(ncols)))  
     } else {
          #return(kronecker(matrix(1, nrow=nrows, ncol=ncols), A))
          #if(is.null(dim(A))) A <- Matrix(A, ncol=1, nrow = length(A))
          
          B <- ((A %*% matrix(1, nrow=nrows, ncol=ncols)))
     }
     #res <- (all.equal(B, repmat(A, nrows, ncols)))
     #if(res != TRUE) warning(res)
     
     return(B)
     
     
     
}

create1Matrix <- function(times) {
     tmp <- rep.int(1, times=times)
     dim(tmp) <- c(1, times)
     return(tmp)
}

benchmark(
     tmp <- create1Matrix(1000),
     tmp2 <- Matrix(1, 1, 1000),
     replications=100
     
     )


# benchmark(
# b1 <- matrix(1, nrow=1, ncol=46568),
# b2 <- Matrix(1, nrow=1, ncol=46568),
# b3 <- fn(46568),
# replications=1000)

# 
# benchmark(
#      tmp1 <- repmat(A, nrows, ncols),
#      tmp2 <- myRepMat(A, nrows, ncols),
#      replications = 100
#      
#      )


# if("gputools" %in% .packages()) {
#      myCrossProd <- gpuCrossprod
#      myTCrossProd <- gpuTcrossprod
#      myMatMult <- gpuMatMult
# } else {
#      myCrossProd <- crossprod
#      myTCrossProd <- tcrossprod
#      myMatMult <- function(x, y) x %*% y
# }

USE_GPU <- "gputools" %in% .packages()
SPARSE_MATRIX_CLASSES <- c("dsCMatrix", "ddiMatrix", "dtCMatrix", "dtTMatrix", "dgTMatrix", "dtRMatrix", "dsTMatrix", "dgRMatrix", "dgCMatrix")

myMatMult <- function(a, b=NULL) {
     # if one is a vector, then no gpu
     # if sparse, then no gpu
     if(is.null(b)) b <- a
     
     if(USE_GPU) {
          dim.a <- dim(a)
          dim.b <- dim(b)
          is.vector <- is.null(dim.a) | is.null(dim.b) | 1 %in% dim.a | 1 %in% dim.b
          if(!is.vector) {
               classes <- c(class(a), class(b))                              
               is.sparse.matrix <- all(classes %in% SPARSE_MATRIX_CLASSES)  # sparseVector as well, but not needed here
          
               if(!is.sparse.matrix) return(gpuMatMult(a, b))
          }
     }
     
     a %*% b
}

myCrossProd <- function(a, b=NULL) {
     # if one is a vector, then no gpu
     # if sparse, then no gpu, unless:
          # if large sparse matrix then gpu; otherwise none
     if(is.null(b)) b <- a
     if(USE_GPU) {
          dim.a <- dim(a)
          dim.b <- dim(b)
          is.vector <- is.null(dim.a) | is.null(dim.b) | 1 %in% dim.a | 1 %in% dim.b
          if(!is.vector) {
               large.matrix <- any(dims > 1000)
               
               classes <- c(class(a), class(b))                              
               is.sparse.matrix <- all(classes %in% SPARSE_MATRIX_CLASSES)  # sparseVector as well, but not needed here
               
               if(!is.sparse.matrix | large.matrix) return(gpuCrossprod(a, b))
          }
     }
     
     crossprod(a, b)
}

myTCrossProd <- function(a, b=NULL) {
     # if one is a vector, then no gpu
     # if sparse, then no gpu
     if(is.null(b)) b <- a
     if(USE_GPU) {
          dim.a <- dim(a)
          dim.b <- dim(b)
          is.vector <- is.null(dim.a) | is.null(dim.b) | 1 %in% dim.a | 1 %in% dim.b
          if(is.vector) {
               classes <- c(class(a), class(b))                              
               is.sparse.matrix <- all(classes %in% SPARSE_MATRIX_CLASSES)  # sparseVector as well, but not needed here
               
               if(!is.sparse.matrix) return(gpuTcrossprod(a, b))
          }
     }
     
     tcrossprod(a, b)
}


# mm <- matrix(runif(3000*2000), 3000, 2000)
# mmsparse<- matrix(c(1,0,0,0,0,0,0,0), nr=3000, nc=2000)
# 
# MM <- Matrix(mm)
# MMsparse<- Matrix(mmsparse)
# 
# 
# benchmark(
#      tmp1 <- myMatMult(mmsparse, t(mmsparse)),              
#      tmp2 <- myMatMult(mm, t(mm)),              
#      tmp3 <- myMatMult(MM, t(MM)),     
#      tmp4 <- myMatMult(MMsparse, t(MMsparse)),    
#      tmp5 <- MM %*% t(MM),
#      tmp6 <- gpuMatMult(MMsparse, t(MMsparse)),
#      replications=10
# )
# 
# benchmark(
#      tmp1 <- myCrossProd(mmsparse),              
#      tmp2 <- myCrossProd(mm),              
#      tmp3 <- myCrossProd(MM),     
#      tmp4 <- myCrossProd(MMsparse),     
#      replications=10
# )
# 
# benchmark(
#      tmp1 <- myTCrossProd(mmsparse),              
#      tmp2 <- myTCrossProd(mm),              
#      tmp3 <- myTCrossProd(MM),     
#      tmp4 <- myTCrossProd(MMsparse),     
#      replications=10
# )


zeros <- function(...) {
     # modified from matlab to incorporate sparse matrix when possible
     
     nargs <- length(dots <- list(...))
     dims <- as.integer(if (nargs == 1 && matlab:::is.size_t(dots[[1]])) {
          dots[[1]]
     } else {
          unlist(dots)
     })
     if (length(dims) == 1) {
          dims[2] <- dims[1]
     }
     if (!(length(dims) > 1)) {
          stop("dimensions must be of length greater than 1")
     }
     else if (!(all(dims > 0))) {
          stop("dimensions must be a positive quantity")
     }
     if(length(dims) == 2) return(Matrix(0, nrow=dims[1], ncol=dims[2], sparse=TRUE))
     #if(length(dims) == 2) return(matrix(0, nrow=dims[1], ncol=dims[2]))
     
     return(array(0, dims))
}

repmat <- function(A, ...) 
{
     nargs <- length(dots <- list(...))
     dims <- as.integer(if (nargs == 1 && matlab:::is.size_t(dots[[1]])) {
          dots[[1]]
     } else {
          unlist(dots)
     })
     if (length(dims) == 1) {
          dims[2] <- dims[1]
     }
     if (!(length(dims) > 1)) {
          stop("dimensions must be of length greater than 1")
     }  else if (!(all(dims > 0))) {
          stop("dimensions must be a positive quantity")
     }
     
     if(is.null(dim(A)) & class(A) == "dsparseVector") A <- Matrix(A, ncol=1, nrow = length(A))
     
     B <- switch(EXPR = mode(A), logical = , complex = , numeric = , S4 = {
          if (all(dims == 1)) {
               A
          } else if (dims[length(dims)] == 1) {
               t(kronecker(array(1, rev(dims)), A))
          } else {
               kronecker(array(1, dims), A)
          }
     }, character = {
          fA <- factor(A, levels = unique(A))
          iA.mat <- Recall(unclass(fA), dims)
          saved.dim <- dim(iA.mat)
          cA.mat <- sapply(seq(along = iA.mat), function(i, A, 
                                                         fac) {
               A[i] <- levels(fac)[A[i]]
          }, iA.mat, fA)
          dim(cA.mat) <- saved.dim
          cA.mat
     }, NULL)
     if (is.null(B)) {
          stop(sprintf("argument %s must be one of [%s]", sQuote("A"), 
                       paste(c("numeric", "logical", "complex", "character"), 
                             collapse = "|")))
     }
     return(B)
}

setMethod(size, signature(X="Matrix", dimen="ANY"), function(X, dimen) if(missing(dimen)) size(as.matrix(X)) else size(as.matrix(X), dimen))



     