# Matrix tests

library(Matrix)
library(gputools)
library(gcbd)


library(rbenchmark)


M <- getMatrix(100)
res1 <- matmultBenchmark(100, 10, trim=0.1)
res2 <- matmultBenchmarkgputools(100,10, trim=0.1)


mm <- matrix(runif(300*200), 300, 200)
mmsparse<- matrix(c(1,0,0,0,0,0,0,0), nr=200, nc=200)

library(ggplot2)

res1 <- sapply(100:1000, matmultBenchmark, n=10, trim=0.1)
resgpu <- sapply(100:1000, matmultBenchmark, n=10, trim=0.1)

dat <- rbind(data.frame(DIM=1:100, TIME=res1, TYPE="CPU"),
             data.frame(DIM=1:100, TIME=resgpu, TYPE="GPU"))

ggplot(dat, aes(x=DIM, y=TIME, colour=TYPE, group=TYPE)) + geom_line()


             



MM <- Matrix(mm)
MMsparse<- Matrix(mmsparse)
lookup<- function(mat){
     for (i in 1:nrow(mat)){
          for (j in 1:ncol(mat)){
               tmp <- mat[i,j]
          }
     }
}

# slightly slower:
# lookup <- function(mat) {
#      i <- expand.grid(row=1:nrow(mat), col=1:ncol(mat))
#      mapply(function(i, j, mat) tmp <- mat[i, j], i=i$row, j=i$col, MoreArgs=list(mat=mat))
# }

benchmark(
     tmp1 <- lookup(mmsparse),
     tmp2 <- lookup(MMsparse),     # several orders of magnitude slower
     tmp3 <- lookup(mm),
     tmp4 <- lookup(MM),           # several orders of magnitude slower
     replications=1)


benchmark(
     tmp1 <- mmsparse %*% mmsparse,               # 10x slower
     tmp2 <- MMsparse %*% MMsparse,               # fastest
     tmp3 <- gpuMatMult(mmsparse, mmsparse),      # second fastest; 10x slower
     tmp4 <- gpuMatMult(MMsparse, MMsparse),      # 10x slower
     replications=100
     )

benchmark(
     tmp1 <- crossprod(mm, mm),                   # 50% slower
     tmp2 <- crossprod(MM, MM),                   # 50% slower
     tmp3 <- gpuCrossprod(mm, mm),                # fastest
     tmp4 <- gpuCrossprod(MM, MM),                # 10% slower
     replications=100
)

benchmark(
     tmp1 <- crossprod(mmsparse, mmsparse),                   # 2.5 times slower
     tmp2 <- crossprod(MMsparse, MMsparse),                   # fastest
     tmp3 <- gpuCrossprod(mmsparse, mmsparse),                # second fastest, about 90% slower
     tmp4 <- gpuCrossprod(MMsparse, MMsparse),                # 2.2 times slower
     replications=100
)


benchmark(
     tmp1 <- tcrossprod(mm, mm),                   # 1.9x slower
     tmp2 <- tcrossprod(MM, MM),                   # 1.9x slower
     tmp3 <- gpuTcrossprod(mm, mm),                # fastest
     tmp4 <- gpuTcrossprod(MM, MM),                # 10% slower
     replications=100
)

benchmark(
     tmp1 <- tcrossprod(mmsparse, mmsparse),                   # 14 times slower
     tmp2 <- tcrossprod(MMsparse, MMsparse),                   # fastest
     tmp3 <- gpuTcrossprod(mmsparse, mmsparse),                # second fastest, about 10x slower
     tmp4 <- gpuTcrossprod(MMsparse, MMsparse),                # 11.5 times slower
     replications=100
)

benchmark(
     tmp1 <- Matrix(mmsparse),          # 18x slower
     tmp2 <- Matrix(mm),                # 7x slower
     tmp3 <- as.matrix(MMsparse),       # fastest (.387 for 1000 reps)
     tmp4 <- as.matrix(MM),             # 14% slower
     replications=1000
     )

# big matrix

mm <- matrix(runif(3000*2000), 3000, 2000)
mmsparse<- matrix(c(1,0,0,0,0,0,0,0), nr=2000, nc=2000)

MM <- Matrix(mm)
MMsparse<- Matrix(mmsparse)

benchmark(
     tmp1 <- mmsparse %*% mmsparse,               # 20x slower
     tmp2 <- MMsparse %*% MMsparse,               # fastest
     tmp3 <- gpuMatMult(mmsparse, mmsparse),      # second fastest; 1.2x slower
     tmp4 <- gpuMatMult(MMsparse, MMsparse),      # 1.3x slower
     replications=10
)

benchmark(
     tmp1 <- crossprod(mm, mm),                   # 23x slower
     tmp2 <- crossprod(MM, MM),                   # 23x slower
     tmp3 <- gpuCrossprod(mm, mm),                # fastest
     tmp4 <- gpuCrossprod(MM, MM),                # 10% slower
     replications=10
)

benchmark(
     tmp1 <- crossprod(mmsparse, mmsparse),                   # 18 times slower
     tmp2 <- crossprod(MMsparse, MMsparse),                   # 5.7 times slower
     tmp3 <- gpuCrossprod(mmsparse, mmsparse),                # fastest
     tmp4 <- gpuCrossprod(MMsparse, MMsparse),                # 8% slower
     replications=10
)


benchmark(
     tmp1 <- tcrossprod(mm, mm),                   # 32x slower
     tmp2 <- tcrossprod(MM, MM),                   # 32x slower
     tmp3 <- gpuTcrossprod(mm, mm),                # fastest
     tmp4 <- gpuTcrossprod(MM, MM),                # 10% slower
     replications=10
)

benchmark(
     tmp1 <- tcrossprod(mmsparse, mmsparse),                   # 30x slower
     tmp2 <- tcrossprod(MMsparse, MMsparse),                   # fastest
     tmp3 <- gpuTcrossprod(mmsparse, mmsparse),                # second fastest, about 4x slower
     tmp4 <- gpuTcrossprod(MMsparse, MMsparse),                # 5.6 times slower (about 10% over tmp3)
     replications=10
)

benchmark(
     tmp1 <- Matrix(mmsparse),          # 23x slower
     tmp2 <- Matrix(mm),                # 5x slower
     tmp3 <- as.matrix(MMsparse),       # 6% slower
     tmp4 <- as.matrix(MM),             # fastest (2.810 for 100 reps)
     replications=100
)

# large matrix

mm <- matrix(runif(30000*2000), 30000, 2000)
mmsparse <- matrix(c(1,0,0,0,0,0,0,0), nr=20000, nc=2000)
mmsparse2 <- matrix(c(1,0,0,0,0,0,0,0), nr=2000, nc=20000)

MM <- Matrix(mm)
MMsparse <- Matrix(mmsparse)
MMsparse2 <-  Matrix(mmsparse2)

benchmark(
     #tmp1 <- mmsparse %*% mmsparse,               # 20x slower
     tmp2 <- MMsparse %*% MMsparse2,               # fastest
     tmp3 <- gpuMatMult(mmsparse, mmsparse2),      # second fastest; 1.2x slower
     tmp4 <- gpuMatMult(MMsparse, MMsparse2),      # 1.3x slower
     replications=1
)

benchmark(
     #tmp1 <- crossprod(mm, mm),                   # 50% slower
     #tmp2 <- crossprod(MM, MM),                   # 50% slower
     tmp3 <- gpuCrossprod(mm, mm),                # fastest
     tmp4 <- gpuCrossprod(MM, MM),                # 10% slower
     replications=2
)

benchmark(
     #tmp1 <- crossprod(mmsparse, mmsparse),                   # 2.5 times slower
     tmp2 <- crossprod(MMsparse, MMsparse),                   # fastest
     tmp3 <- gpuCrossprod(mmsparse, mmsparse),                # second fastest, about 90% slower
     tmp4 <- gpuCrossprod(MMsparse, MMsparse),                # 2.2 times slower
     replications=2
)


benchmark(
     #tmp1 <- tcrossprod(mm, mm),                   # 1.9x slower
     #tmp2 <- tcrossprod(MM, MM),                   # 1.9x slower
     tmp3 <- gpuTcrossprod(mm, mm),                # fastest
     tmp4 <- gpuTcrossprod(MM, MM),                # 10% slower
     replications=2
)

benchmark(
     #tmp1 <- tcrossprod(mmsparse, mmsparse),                  # 14 times slower
     tmp2 <- tcrossprod(MMsparse, MMsparse),                   # fastest
     tmp3 <- gpuTcrossprod(mmsparse, mmsparse),                # second fastest, about 10x slower
     tmp4 <- gpuTcrossprod(MMsparse, MMsparse),                # 11.5 times slower
     replications=2
)

benchmark(
     tmp1 <- Matrix(mmsparse),
     tmp2 <- Matrix(mm),
     tmp3 <- as.matrix(MMsparse),
     tmp4 <- as.matrix(MM),
     replications=1
)



# Vector * matrix

# small
mm <- matrix(runif(300*200), 300, 200)
mmsparse<- matrix(c(1,0,0,0,0,0,0,0), nr=200, nc=200)
vec <- runif(200)
vecsparse <- sample(c(1,0), size=200, replace=T, prob=c(.1, .9))
dim(vec) <- c(1, 200)
dim(vecsparse) <- c(1, 200)

MM <- Matrix(mm)
MMsparse<- Matrix(mmsparse)

benchmark(
     tmp1 <- vecsparse %*% mmsparse,               # fastest
     tmp2 <- vecsparse %*% MMsparse,               # 2.1x slower
     tmp3 <- gpuMatMult(mmsparse, vecsparse),      # 13x slower
     tmp4 <- gpuMatMult(MMsparse, vecsparse),      # 15.2x slower
     tmp5 <- vec %*% MMsparse,                     # second fastest; 2x slower
     tmp6 <- gpuMatMult(MMsparse, vec),            # 15.4x slower
     replications=1000
)

benchmark(
     tmp1 <- crossprod(vecsparse, mmsparse),                # fastest
     tmp2 <- crossprod(vecsparse, MMsparse),                # 4.7x slower
     tmp3 <- gpuCrossprod(vecsparse, mmsparse),             # 33x slower
     tmp4 <- gpuCrossprod(vecsparse, MMsparse),             # 39x slower
     tmp5 <- crossprod(vec, MMsparse),                      # 4.8x slower
     tmp6 <- gpuCrossprod(vec, MMsparse),                   # 39x slower
     replications=1000
)

benchmark(
     tmp1 <- tcrossprod(vecsparse, mmsparse),                # fastest
     tmp2 <- tcrossprod(vecsparse, MMsparse),                # 4.3x slower
     tmp3 <- gpuTcrossprod(vecsparse, mmsparse),             # 14.6 slower
     tmp4 <- gpuTcrossprod(vecsparse, MMsparse),             # 17.7 slower
     tmp5 <- tcrossprod(vec, MMsparse),                      # 10.7x slower
     tmp6 <- tcrossprod(vec, mmsparse),                     # 1.6% slower
     tmp7 <- gpuTcrossprod(vec, MMsparse),                   # 18.9x slower
     replications=1000   
)



# big
mm <- matrix(runif(3000*2000), 3000, 2000)
mmsparse<- matrix(c(1,0,0,0,0,0,0,0), nr=2000, nc=2000)
vec <- runif(2000)
vecsparse <- sample(c(1,0), size=2000, replace=T, prob=c(.1, .9))
dim(vec) <- c(1, 2000)
dim(vecsparse) <- c(1, 2000)

MM <- Matrix(mm)
MMsparse<- Matrix(mmsparse)


benchmark(
     tmp1 <- vecsparse %*% mmsparse,               
     tmp2 <- vecsparse %*% MMsparse,              
     tmp3 <- gpuMatMult(vecsparse, mmsparse),    
     tmp4 <- gpuMatMult(vecsparse, MMsparse),     
     tmp5 <- vec %*% MMsparse,                    # fastest
     tmp6 <- gpuMatMult(vec, MMsparse),            
     replications=100
)
# test replications elapsed relative user.self sys.self user.child sys.child
# 1          tmp1 <- vecsparse %*% mmsparse          100   1.423    8.730     1.423    0.000          0         0
# 2          tmp2 <- vecsparse %*% MMsparse          100   0.169    1.037     0.167    0.003          0         0
# 3 tmp3 <- gpuMatMult(vecsparse, mmsparse)          100   9.548   58.577     7.085    2.488          0         0
# 4 tmp4 <- gpuMatMult(vecsparse, MMsparse)          100  18.123  111.184    14.415    3.750          0         0
# 5                tmp5 <- vec %*% MMsparse          100   0.163    1.000     0.159    0.004          0         0
# 6       tmp6 <- gpuMatMult(vec, MMsparse)          100  17.950  110.123    14.277    3.713          0         0


benchmark(
     tmp1 <- crossprod(t(vecsparse), MMsparse),            # fastest
     tmp2 <- crossprod(t(vecsparse), MMsparse),             # second fastest       
     tmp3 <- gpuCrossprod(t(vecsparse), mmsparse),             
     tmp4 <- gpuCrossprod(t(vecsparse), MMsparse),            
     tmp5 <- crossprod(t(vec), MMsparse),                    
     tmp6 <- gpuCrossprod(t(vec), MMsparse),                  
     replications=100
)

# test replications elapsed relative user.self sys.self user.child sys.child
# 1    tmp1 <- crossprod(t(vecsparse), MMsparse)          100   0.157    1.000     0.156    0.001          0         0
# 2    tmp2 <- crossprod(t(vecsparse), MMsparse)          100   0.165    1.051     0.164    0.001          0         0
# 3 tmp3 <- gpuCrossprod(t(vecsparse), mmsparse)          100  15.299   97.446    12.870    2.461          0         0
# 4 tmp4 <- gpuCrossprod(t(vecsparse), MMsparse)          100  17.907  114.057    14.450    3.493          0         0
# 5          tmp5 <- crossprod(t(vec), MMsparse)          100   0.178    1.134     0.176    0.002          0         0
# 6       tmp6 <- gpuCrossprod(t(vec), MMsparse)          100  18.078  115.146    14.498    3.640          0         0



benchmark(
     tmp1 <- tcrossprod(vecsparse, mmsparse),               # third fastest (4.4x slower) 
     tmp2 <- tcrossprod(vecsparse, MMsparse),               # fastest
     tmp3 <- gpuTcrossprod(vecsparse, mmsparse),             
     tmp4 <- gpuTcrossprod(vecsparse, MMsparse),             
     tmp5 <- tcrossprod(vec, MMsparse),                    # second fastest (45% slower)
     tmp6 <- tcrossprod(vec, mmsparse),                    
     tmp7 <- gpuTcrossprod(vec, MMsparse),                  
     tmp8 <- tcrossprod(vecsparse, MM),
     replications=100
)

# test replications elapsed relative user.self sys.self user.child sys.child
# 1    tmp1 <- tcrossprod(vecsparse, mmsparse)          100   3.505    4.392     3.505    0.001          0         0
# 2    tmp2 <- tcrossprod(vecsparse, MMsparse)          100   0.798    1.000     0.798    0.001          0         0
# 3 tmp3 <- gpuTcrossprod(vecsparse, mmsparse)          100  15.002   18.799    12.596    2.434          0         0
# 4 tmp4 <- gpuTcrossprod(vecsparse, MMsparse)          100  17.824   22.336    14.272    3.603          0         0
# 5          tmp5 <- tcrossprod(vec, MMsparse)          100   1.161    1.455     0.945    0.218          0         0
# 6          tmp6 <- tcrossprod(vec, mmsparse)          100   3.492    4.376     3.489    0.005          0         0
# 7       tmp7 <- gpuTcrossprod(vec, MMsparse)          100  18.051   22.620    14.413    3.692          0         0
# 8          tmp8 <- tcrossprod(vecsparse, MM)          100   4.904    6.145     4.901    0.006          0         0

# Rules for when to use each:
# Indexing
#    Never index with Matrix()
# Conversions
#    converting using as.matrix in the gpu functions costs about 10% time
#    as.matrix is much faster than Matrix() for conversions.  
# Matrix Multiply
#    gpu is basically always better when multiplying dense matrices
#    do not use gpu to multiply vector with a matrix
#    100 x 100 matrices: sparseMatrix is 10x better than gpu
#    1000 x 1000 matrices: sparseMatrix is ~ 20% better than gpu 
#    10000 x 10000 matrices: 
# Crossprod
#    gpu is basically always better when cross-multiplying dense matrices
#    do not use gpu to multiply vector with a matrix
#    100 x 100 matrices: gpu is 90% slower on sparseMatrix
#    1000 x 1000 matrices: gpu is fastest; sparseMatrix is 5x slower
#    10000 x 10000 matrices: 
# Tcrossprod
#    gpu is basically always better when cross-multiplying dense matrices
#    do not use gpu to multiply vector with a matrix
#    100 x 100 matrices: gpu is 10x slower on sparseMatrix
#    1000 x 1000 matrices: gpu is 4x slower than sparseMatrix
#    10000 x 10000 matrices: 

# never index without converting to as.matrix()
# Always use gpu with dense matrices; never use gpu with vector * matrix
# Matrix only sees speed-up with sparse matrices; otherwise basically same as normal matrix functions
# Sparsematrix: 
#    use Matrix to matrix multiply for small or big matrices
#    use Matrix to crossprod for small matrices
#    use Matrix to tcrossprod for small or big matrices
#    use Matrix when multiplying vectors with sparse matrix


# Using repmat

# repmat (A, m)
# repmat (A, m, n) 
# Form a block matrix of size m by n, with a copy of matrix A as each element. If n is not specified, form an m by m block matrix.
# replicate and tile a matrix to the number of dimensions.  

# why not use rbind and cbind?


myRepMat1 <- function(A, ...) {
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
     } else if (!(all(dims > 0))) {
          stop("dimensions must be a positive quantity")
     }
     
     # dims[2] is number of columns * original dimensions; dims[1] is number of rows
     if(all(dims == 1)) return(A)
     
     if(length(dims) == 2) {
          B <- NULL
          if(dims[2] > 1) {
               B <- do.call(cBind, rep(list(A), times=dims[2]))
               if(dims[1] > 1) B <- do.call(rBind, rep(list(B), times=dims[1]))
          } else {
               B <- do.call(rBind, rep(list(A), times=dims[1]))
               
          }
          return(B)
     }
     
     return(repmat(A, ...))
     
}

# apparently cBind is slow!

A <- matrix(1:3, 300, 400)
A2 <- 1:100




benchmark(
     tmp1 <- repmat(A1, 20, 40),
     tmp2 <- myRepMat1(A1, 20, 40),
     tmp3 <- repmat(A2, 1, 20),
     tmp4 <- myRepMat1(A2, 1, 20),
     replications=10
     )


benchmark(
     B1 <- do.call(rBind, rep(list(A2), times=5)),
     B2 <- do.call(rBind, replicate(5, {A2}, simplify=F)),
     B3 <- kronecker(array(1, c(1, 20)), A2)
     replications = 100
     )

tA <- t(A)
benchmark(
     B1 <- do.call(cBind, rep(list(A), times=5)),
     B2 <- do.call(cBind, replicate(5, {A}, simplify=F)),
     B3 <- t(replicate(20, {tA}, simplify=T)),
     replications = 100
)


myRepMat <- function(A, ...) {
     # only works for numeric matrices with set number of rows & columns
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
     } else if (!(all(dims > 0))) {
          stop("dimensions must be a positive quantity")
     }
     
     # dims[2] is number of columns * original dimensions; dims[1] is number of rows
     B <- switch(EXPR = mode(A), logical = , complex = , numeric = , S4 = {
          if (all(dims == 1)) {
               A
          } else if(length(dims) == 2) { 
               # working with a matrix
               if(dims[2] == 1) {
                    #B <- (t(myMatMult(A, create1Matrix(nrows))))
                    B <- t(myMatMult(A, rep.int(1, times=dims[1])))
               } else if(dims[1] == 1) {
                    #B <- (myMatMult(A, create1Matrix(ncols)))  
                    B <- myMatMult(A, rep.int(1, times=dims[2]))
               } else {
                    #return(kronecker(matrix(1, nrow=nrows, ncol=ncols), A))
                    B <- (myMatMult(A, matrix(1, nrow=dims[1], ncol=dims[2])))
               }
               
          } else if (dims[length(dims)] == 1) {  # last dimension is a 1
               
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

mm <- matrix(runif(3000*2000), 3000, 2000)
mmsparse<- matrix(c(1,0,0,0,0,0,0,0), nr=2000, nc=2000)

MM <- Matrix(mm)
MMsparse<- Matrix(mmsparse)


benchmark(
     tmp1 <- repmat(mm, 1, 20),
     tmp2 <- myRepMat(mm, 1, 20),
     replications=5)
     
benchmark(
     tmp2 <- repmat(MM, 1, 20),
     tmp3 <- repmat(mmsparse, 1, 20),
     tmp4 <- repmat(MMsparse, 1, 20),
     replications = 5
     )

