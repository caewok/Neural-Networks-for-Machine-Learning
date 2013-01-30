# Run perceptron for each data set

source("Perceptron Learning Algorithm/ImportDatasets.R")
source("Perceptron Learning Algorithm/LearnPerceptron.R")

with(dataset4, 
     learn_perceptron(neg.examples.nobias, pos.examples.nobias, w.init, w.gen.feas)
)

learn_perceptron(dataset1$neg.examples.nobias, dataset1$pos.examples.nobias, dataset1$w.init, dataset1$w.gen.feas)

fn <- function(x) nrow(x)

fn(dataset1$neg.examples.nobias)