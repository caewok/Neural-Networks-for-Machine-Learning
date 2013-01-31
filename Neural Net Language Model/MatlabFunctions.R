# Necessary matlab functions

# Matlab helper functions
# Functions to replicate various matlab commands

# most are found in the matlab library
library(matlab)
library(gputools)
library(Matrix)

# HELPER FUNCTIONS
randn <- function(x, y) matrix(data=rnorm(x*y), nrow=x, ncol=y)
myPrintf <- function(txt, ...) writeLines(sprintf(txt, ...), sep="", con=stdout(), useBytes=TRUE)

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
               large.matrix <- any(dim.a > 1000) | any(dim.b > 1000)
               
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
          if(!is.vector) {
               classes <- c(class(a), class(b))                              
               is.sparse.matrix <- all(classes %in% SPARSE_MATRIX_CLASSES)  # sparseVector as well, but not needed here
               
               if(!is.sparse.matrix) return(gpuTcrossprod(a, b))
          }
     }
     
     tcrossprod(a, b)
}




