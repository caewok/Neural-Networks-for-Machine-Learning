# Necessary matlab functions

# Matlab helper functions
# Functions to replicate various matlab commands

# most are found in the matlab library
library(matlab)

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