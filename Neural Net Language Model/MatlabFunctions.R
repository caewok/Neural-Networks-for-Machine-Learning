# Necessary matlab functions

# Matlab helper functions
# Functions to replicate various matlab commands

# most are found in the matlab library
library(matlab)

# HELPER FUNCTIONS
randn <- function(x, y) matrix(data=rnorm(x*y), nrow=x, ncol=y)
myPrintf <- function(txt, ...) writeLines(sprintf(txt, ...), sep="", con=stdout(), useBytes=TRUE)

