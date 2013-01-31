# Original training file, with minor modifications to run in R

# setwd("~/R_Projects/Neural Networks for Machine Learning")
#library(gputools)
#library(Matrix)
#library(parallel); options(mc.cores=4)
# library(R.utils)
# source("Neural Net Language Model/Train.R")

source("Neural Net Language Model/MatlabFunctions.R")
source("Neural Net Language Model/LoadData.R")
source("Neural Net Language Model/FProp.R")


# see also http://mathesaurus.sourceforge.net/octave-r.html

# % This function trains a neural network language model.
train <- function(epochs) {
     #      % Inputs:
     #      %   epochs: Number of epochs to run.
     #      % Output:
     #      %   model: A struct containing the learned weights and biases and vocabulary.
     
     start_time <- proc.time() 
     
     #      % SET HYPERPARAMETERS HERE.
     batchsize = 1000  # Mini-batch size; default = 100.
     learning_rate = 0.1  # Learning rate; default = 0.1.
     momentum = 0.9  # Momentum; default = 0.9.
     numhid1 = 50  # Dimensionality of embedding space; default = 50.
     numhid2 = 200  # Number of units in hidden layer; default = 200.
     init_wt = 0.01  # Standard deviation of the normal distribution, which is sampled to get the initial weights; default = 0.01
     
     #      % VARIABLES FOR TRACKING TRAINING PROGRESS.
     # number is the number of batches to run before showing training & validation
     show_training_CE_after = 100
     show_validation_CE_after = 1000
     
     #      % LOAD DATA.
     data <- load_data(batchsize)
     tmp <- size(data$train_input[[1]])
     numwords <- tmp[1]
     numbatches <- length(data$train_input)
     vocab_size <- size(data$vocab, 2)  
     
     
     # Initialize weights and gradients
     d1 <- list(word_embedding=vocab_size,
                embed_to_hid=numwords * numhid1,
                hid_to_output=numhid2,
                hid_bias=numhid2,
                output_bias=vocab_size)
     d2 <- list(word_embedding=numhid1,
                embed_to_hid=numhid2,
                hid_to_output=vocab_size,
                hid_bias=1,
                output_bias=1)
     
     
     weights <- c(mapply(function(x, y, init_wt) init_wt*randn(x,y), x=d1[1:3], y=d2[1:3], MoreArgs=list(init_wt=init_wt)),
                  mapply(zeros, d1[4:5], d2[4:5])) # mapply(rep.int, times=d1[4:5], MoreArgs=list(x=0))
     
     deltas <- mapply(zeros, d1[1:5], d2[1:5])
     
#      gradients <- list(word_embedding=zeros(d1[[1]], d2[[1]]),
#                        embed_to_hid=NULL,
#                        hid_to_output=NULL,
#                        hid_bias=NULL,
#                        output_bias=NULL)
     
     expansion_matrix = eye(vocab_size);
     count = 0
     
     
     
     #                %% Expand the target to a sparse 1-of-K vector.
     #      expanded_target_batch = expansion_matrix[, as.integer(target_batch)]   # could expand this in a list outside the loop
     expanded_target_batches <- lapply(data$train_target, function(target_batch, expansion_matrix) {expansion_matrix[, as.integer(target_batch)]}, expansion_matrix=expansion_matrix)
     # mclapply seems to do worse
     
     expandInputBatchFn <- function(w, expansion_matrix, input_batch) {
          expansion_matrix[, as.integer(input_batch[w,])]
          
     }
     expandInputBatchFn2 <- function(numwords, expansion_matrix, input_batch) {
          lapply(1:numwords, expandInputBatchFn, expansion_matrix=expansion_matrix, input_batch=input_batch)
     }
     
     input_batch_expansions <- lapply(data$train_input, expandInputBatchFn2, numwords=numwords, expansion_matrix=expansion_matrix)
     
     expanded_valid_target <- expansion_matrix[, as.integer(data$valid_target)]
     
     
     
     #      % TRAIN.
     for(epoch in 1:epochs) {
          myPrintf('Epoch %d\n', epoch)
          this_chunk_CE <- 0
          trainset_CE <- 0
          
         
          #           % LOOP OVER MINI-BATCHES.
          for(m in 1:numbatches) {
               input_batch <- data$train_input[[m]]
               target_batch <- data$train_target[[m]]
               expanded_target_batch <- expanded_target_batches[[m]]
               # dim(target_batch) <- NULL
               
               #                % FORWARD PROPAGATE.
               #                % Compute the state of each layer in the network given the input batch and all weights and biases
               # returns the embedding, hidden and output layer states
               
               
               neural_net_states <- fprop(input_batch, weights, fn)
               
               res <- microbenchmark(
                    fprop(input_batch, weights, reshape),
                    fprop(input_batch, weights, myReshape),
                    times=100
                    )
               print(res, "s")
               boxplot(res, "s")
               
               
               
               #                % MEASURE LOSS FUNCTION.
               CE = CEfn(expanded_target_batch, neural_net_states$output_layer_state, batchsize)
               count =  count + 1
               this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count
               trainset_CE = trainset_CE + (CE - trainset_CE) / m
               #printf('\rBatch %d Train CE %.3f', m, this_chunk_CE)
               myPrintf('\rBatch %d Train CE %.3f', m, this_chunk_CE)
               if (mod(m, show_training_CE_after) == 0) {
                    myPrintf('\n')
                    count = 0
                    this_chunk_CE = 0
               }
               
               #                % BACK PROPAGATE.
               gradients <- BProp(neural_net_states, weights, expanded_target_batch, input_batch_expansions[[m]], numhid1, numwords)
               
               #                % UPDATE WEIGHTS AND BIASES.  
               deltas <- mapply(delta_update_fn, delta=deltas, gradient=gradients, MoreArgs=list(momentum=momentum, batchsize=batchsize))
               weights <- mapply(weights_update_fn, delta=deltas, weight=weights, MoreArgs=list(learning_rate=learning_rate))  
               
               #                % VALIDATE.
               if(mod(m, show_validation_CE_after) == 0) {
                    CE <- Validate(data$valid_input, weights, expanded_valid_target)
                    myPrintf(' Validation CE %.3f\n', CE)
               }
          }          
          myPrintf('\rAverage Training CE %.3f\n', trainset_CE)
     }
     myPrintf('Finished Training.\n')
     myPrintf('Final Training CE %.3f\n', trainset_CE)
     
     #      % EVALUATE ON VALIDATION SET.
     CE <- Validate(data$valid_input, weights, expanded_valid_target)
     myPrintf('\rFinal Validation CE %.3f\n', CE)
     
     #      % EVALUATE ON TEST SET.
     myPrintf('\rRunning test ...');
     CE <- Validate(data$test_input, weights, expansion_matrix[, as.integer(data$test_target)])
     myPrintf('\rFinal Test CE %.3f\n', CE)
     
     end_time <- proc.time() 
     
     print(end_time - start_time)
     return(list(weights=weights,
                 vocab=data$vocab))
}

delta_update_fn <- function(delta, gradient, momentum, batchsize) { momentum * delta + gradient / batchsize}
weights_update_fn <- function(delta, weight, learning_rate) {weight - (learning_rate * delta)}

BProp <- function(neural_net_states, weights, expanded_target_batch, input_batch_expansion, numhid1, numwords) {
     # input_batch_expansion should be list of 3
     gradients <- vector("list", 5)
     names(gradients) <- names(weights)
     
     #                % COMPUTE DERIVATIVE.
     
     #                %% Compute derivative of cross-entropy loss function.
     error_deriv = neural_net_states$output_layer_state - expanded_target_batch  # all the derivatives are not saved between loops
     gradients$hid_to_output <- neural_net_states$hidden_layer_state %*% t(error_deriv)
     gradients$output_bias <- rowSums(error_deriv)
     back_propagated_deriv_1 <- (weights$hid_to_output %*% error_deriv) * neural_net_states$hidden_layer_state * (1 - neural_net_states$hidden_layer_state)
     gradients$embed_to_hid <- neural_net_states$embedding_layer_state %*% t(back_propagated_deriv_1)
     gradients$hid_bias <- rowSums(back_propagated_deriv_1)
     back_propagated_deriv_2 <- weights$embed_to_hid %*% back_propagated_deriv_1
     
     # Embedding layer
     
     t_back_prop_derivs <- lapply(1:numwords, 
                                  function(w, back_propagated_deriv_2, numhid1) { t(back_propagated_deriv_2[(1 + (w - 1) * numhid1):(w * numhid1), ]) },
                                  back_propagated_deriv_2=back_propagated_deriv_2,
                                  numhid1=numhid1)
     
     mult_res <- mapply("%*%", input_batch_expansion, t_back_prop_derivs, SIMPLIFY=F)
     gradients$word_embedding <- mult_res[[1]] + mult_res[[2]] + mult_res[[3]]
     
     return(gradients)
     
     
}


Validate <- function(input, weights, expanded_target) {
     myPrintf('\rRunning validation ...')
     datasetsize = size(input, 2);
     neural_net_states <- fprop(input, weights) 
     return(CEfn(expanded_target, neural_net_states$output_layer_state, datasetsize))
}

CEfn <- function(expanded_target, output_layer_state, datasetsize) {
     tiny <- exp(-30)
     CE <- -matlab::sum(matlab::sum(expanded_target * log(output_layer_state + tiny))) / datasetsize
     return(CE)
}

# Rprof()
# model <- train(1)
# Rprof(NULL)
# summaryRprof()
