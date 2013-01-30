# Original training file, with minor modifications to run in R

# setwd("~/R_Projects/Neural Networks for Machine Learning")
#library(gputools)
#library(Matrix)
#library(parallel); options(mc.cores=4)
# library(R.utils)

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
     tmp <- size(data$train_input[[1]]$value)
     numwords <- tmp[1]
     #batchsize <- tmp[2]
     #numbatches <- tmp[3]
     numbatches <- length(data$train_input)
     vocab_size <- size(data$vocab, 2)  
     
     word_embedding_weights = init_wt * randn(vocab_size, numhid1);
     embed_to_hid_weights = init_wt * randn(numwords * numhid1, numhid2);
     hid_to_output_weights = init_wt * randn(numhid2, vocab_size);
     hid_bias = zeros(numhid2, 1);
     output_bias = zeros(vocab_size, 1);
     
     word_embedding_weights_delta = zeros(vocab_size, numhid1);
     word_embedding_weights_gradient = zeros(vocab_size, numhid1);
     embed_to_hid_weights_delta = zeros(numwords * numhid1, numhid2);
     hid_to_output_weights_delta = zeros(numhid2, vocab_size);
     hid_bias_delta = zeros(numhid2, 1);
     output_bias_delta = zeros(vocab_size, 1);
     expansion_matrix = eye(vocab_size);
     count = 0
     tiny = exp(-30)
         
     #      % TRAIN.
     for(epoch in 1:epochs) {
          myPrintf('Epoch %d\n', epoch)
          this_chunk_CE <- 0
          trainset_CE <- 0
          
          #inputIT <- ihasNext(isplitCols(data$train_input, chunkSize=batchsize))
          #targetIT <- ihasNext(isplitCols(data$train_target, chunkSize=batchsize))
          #m <- 0
          #           % LOOP OVER MINI-BATCHES.
          for(m in 1:numbatches) {
           #while(hasNext(inputIT) & hasNext(targetIT)) {    
                
                
               #input_batch <- data$train_input[,,m]
               #target_batch <- data$train_target[,,m]
               #input_batch <- nextElem(inputIT)
               #target_batch <- nextElem(targetIT)
               input_batch <- data$train_input[[m]]$value
               target_batch <- data$train_target[[m]]$value
               
               dim(target_batch) <- NULL
               
               #                % FORWARD PROPAGATE.
               #                % Compute the state of each layer in the network given the input batch and all weights and biases
               # returns the embedding, hidden and output layer states
               neural_net_states <- fprop(input_batch, word_embedding_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
               
               #                benchmark(
               #                     neural_net_states <- fprop(input_batch, weights, repmat),
               #                     neural_net_states2 <- fprop(input_batch, weights, myRepMat),
               #                     neural_net_states3 <- fprop(input_batch, weights, myRepMat2),
               #                     neural_net_states3 <- fprop(input_batch, weights, myRepMat3),
               #                     neural_net_states4 <- fprop(input_batch, weights, myRepMat4),
               #                     replications=10
               #                     )
               
               #                % COMPUTE DERIVATIVE.
               #                %% Expand the target to a sparse 1-of-K vector.
               expanded_target_batch = expansion_matrix[, as.integer(target_batch)]
               #                %% Compute derivative of cross-entropy loss function.
               error_deriv = neural_net_states$output_layer_state - expanded_target_batch
               
               #                % MEASURE LOSS FUNCTION.
               CE = -matlab::sum(matlab::sum(expanded_target_batch * log(neural_net_states$output_layer_state + tiny))) / batchsize
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
               #                %% OUTPUT LAYER.
               # tcrossprod == x %*% t(y)
               #gradients$hid_to_output =  myTCrossProd(neural_net_states$hidden_layer_state, error_deriv)     
               #gradients$output_bias = rowSums(error_deriv)
               #back_propagated_deriv_1 = myMatMult(weights$hid_to_output,error_deriv) * neural_net_states$hidden_layer_state * (1 - neural_net_states$hidden_layer_state)
               hid_to_output_weights_gradient <- neural_net_states$hidden_layer_state %*% t(error_deriv)
               output_bias_gradient <- rowSums(error_deriv)
               back_propagated_deriv_1 <- (hid_to_output_weights %*% error_deriv) * neural_net_states$hidden_layer_state * (1 - neural_net_states$hidden_layer_state)
               
               
               #                %% HIDDEN LAYER.
               #                % FILL IN CODE. Replace the line below by one of the options.
               #                gradients$embed_to_hid = zeros(numhid1 * numwords, numhid2);
               #                % Options:
               #                     % (a) embed_to_hid_weights_gradient = t(back_propagated_deriv_1) %*% embedding_layer_state;
               #gradients$embed_to_hid = myTCrossProd(neural_net_states$embedding_layer_state, back_propagated_deriv_1);
               embed_to_hid_weights_gradient <- neural_net_states$embedding_layer_state %*% t(back_propagated_deriv_1)
               #                % (c) embed_to_hid_weights_gradient = back_propagated_deriv_1;
               #                % (d) embed_to_hid_weights_gradient = embedding_layer_state;
               
               #                % FILL IN CODE. Replace the line below by one of the options.
               #                gradients$hid_bias = zeros(numhid2, 1);
               #                % Options
               #gradients$hid_bias = rowSums(back_propagated_deriv_1);
               hid_bias_gradient <- rowSums(back_propagated_deriv_1)
               #                % (b) hid_bias_gradient = apply(back_propagated_deriv_1, 2, sum);
               #                % (c) hid_bias_gradient = back_propagated_deriv_1;
               #                % (d) hid_bias_gradient = back_propagated_deriv_1';
               
               #                % FILL IN CODE. Replace the line below by one of the options.
               #                 back_propagated_deriv_2 = zeros(numhid2, batchsize);
               #                % Options
               #back_propagated_deriv_2 = myMatMult(weights$embed_to_hid, back_propagated_deriv_1);
               back_propagated_deriv_2 <- embed_to_hid_weights %*% back_propagated_deriv_1
               #                % (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
               #                % (c) back_propagated_deriv_2 = back_propagated_deriv_1' * embed_to_hid_weights;
               #                % (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights';
               
               #gradients$word_embedding[,] = 0;
               #                %% EMBEDDING LAYER.
               
               word_embedding_weights_gradient[,] <- 0
               for(w in 1:numwords) {
                    #tmp <- back_propagated_deriv_2[(1 + (w - 1) * numhid1):(w * numhid1), ]
                    #gradients$word_embedding= gradients$word_embedding + myTCrossProd(expansion_matrix[, as.integer(input_batch[w, ])], tmp)
                    word_embedding_weights_gradient <- word_embedding_weights_gradient + 
                         expansion_matrix[,as.integer(input_batch[w,])] %*% 
                         t(back_propagated_deriv_2[(1 + (w - 1) * numhid1):(w * numhid1), ])
               }
               
             
               #                % UPDATE WEIGHTS AND BIASES.
               
               word_embedding_weights_delta = momentum * word_embedding_weights_delta +
                    word_embedding_weights_gradient / batchsize;
               word_embedding_weights = word_embedding_weights - (learning_rate * word_embedding_weights_delta);
               
               embed_to_hid_weights_delta = momentum * embed_to_hid_weights_delta + 
                    embed_to_hid_weights_gradient / batchsize;
               embed_to_hid_weights = embed_to_hid_weights - (learning_rate * embed_to_hid_weights_delta);
               
               hid_to_output_weights_delta = momentum * hid_to_output_weights_delta + 
                    hid_to_output_weights_gradient / batchsize;
               hid_to_output_weights = hid_to_output_weights - (learning_rate * hid_to_output_weights_delta);
               
               hid_bias_delta = momentum * hid_bias_delta + 
                    hid_bias_gradient / batchsize;
               hid_bias = hid_bias - (learning_rate * hid_bias_delta);
               
               output_bias_delta = momentum * output_bias_delta + 
                    output_bias_gradient / batchsize;
               output_bias = output_bias - (learning_rate * output_bias_delta);
               
               #                % VALIDATE.
               if(mod(m, show_validation_CE_after) == 0) {
                    myPrintf('\rRunning validation ...')
                    neural_net_states <- fprop(data$valid_input, word_embedding_weights, embed_to_hid_weights,
                                               hid_to_output_weights, hid_bias, output_bias)
                    #                     benchmark(
                    #                     neural_net_states <- fprop(data$valid_input, weights, fn=repmat),
                    #                     neural_net_states2 <- fprop(data$valid_input, weights, fn=myRepMat),
                    #                     neural_net_states3 <- fprop(data$valid_input, weights, myRepMat2),
                    #                     neural_net_states4 <- fprop(data$valid_input, weights, myRepMat3),
                    #                     neural_net_states5 <- fprop(data$valid_input, weights, myRepMat4),
                    #                     replications=2
                    #                     )
                    
                    datasetsize <- size(data$valid_input, 2)
                    expanded_valid_target <- expansion_matrix[, as.integer(data$valid_target)]
                    CE = -matlab::sum(matlab::sum(expanded_valid_target * log(neural_net_states$output_layer_state + tiny))) /datasetsize
                    myPrintf(' Validation CE %.3f\n', CE)
               }
          }          
          myPrintf('\rAverage Training CE %.3f\n', trainset_CE)
     }
     myPrintf('Finished Training.\n')
     myPrintf('Final Training CE %.3f\n', trainset_CE)
     
     #      % EVALUATE ON VALIDATION SET.
     myPrintf('\rRunning validation ...')
     
     neural_net_states <- fprop(data$valid_input, word_embedding_weights, embed_to_hid_weights,
                                hid_to_output_weights, hid_bias, output_bias)
     datasetsize <- size(data$valid_input, 2)
     expanded_valid_target <- expansion_matrix[, as.integer(data$valid_target)]
     CE = -matlab::sum(matlab::sum(expanded_valid_target * log(neural_net_states$output_layer_state + tiny))) /datasetsize
     myPrintf('\rFinal Validation CE %.3f\n', CE)
     
     #      % EVALUATE ON TEST SET.
     myPrintf('\rRunning test ...');
     
     neural_net_states <- fprop(data$test_input, word_embedding_weights, embed_to_hid_weights,
                                hid_to_output_weights, hid_bias, output_bias)
     datasetsize = size(data$valid_input, 2);
     expanded_valid_target = expansion_matrix[, as.integer(data$test_target)];
     CE = -matlab::sum(matlab::sum(expanded_valid_target * log(neural_net_states$output_layer_state + tiny))) / datasetsize
     myPrintf('\rFinal Test CE %.3f\n', CE)
     
     end_time <- proc.time() 
     
     print(end_time - start_time)
     return(list(word_embedding_weights = word_embedding_weights, 
                 embed_to_hid_weights = embed_to_hid_weights,
                 hid_to_output_weights = hid_to_output_weights,
                 hid_bias = hid_bias,
                 output_bias = output_bias,
                 vocab=data$vocab))
}



# Rprof()
# model <- train(1)
# Rprof(NULL)
# summaryRprof()
