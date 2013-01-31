# Original FProp file, with minor modifications to accomodate R

# library(Matrix)
#library(matrixStats)
library(R.matlab)
library(matrixStats)

fprop <- function(input_batch, weights) {
     #      % This method forward propagates through a neural network.
     #      % Inputs:
     #      %   input_batch: The input data as a matrix of size numwords X batchsize where,
     #      %     numwords is the number of words, batchsize is the number of data points.
     #      %     So, if input_batch(i, j) = k then the ith word in data point j is word
     #      %     index k of the vocabulary.
     #      %
     #      %   word_embedding_weights: Word embedding as a matrix of size
     #      %     vocab_size X numhid1, where vocab_size is the size of the vocabulary
     #      %     numhid1 is the dimensionality of the embedding space.
     #      %
     #      %   embed_to_hid_weights: Weights between the word embedding layer and hidden
     #      %     layer as a matrix of size numhid1*numwords X numhid2, numhid2 is the
     #      %     number of hidden units.
     #      %
     #      %   hid_to_output_weights: Weights between the hidden layer and output softmax
     #      %               unit as a matrix of size numhid2 X vocab_size
     #      %
     #      %   hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
     #      %
     #      %   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
     #      %
     #      % Outputs:
     #      %   embedding_layer_state: State of units in the embedding layer as a matrix of
     #      %     size numhid1*numwords X batchsize
     #      %
     #      %   hidden_layer_state: State of units in the hidden layer as a matrix of size
     #      %     numhid2 X batchsize
     #      %
     #      %   output_layer_state: State of units in the output layer as a matrix of size
     #      %     vocab_size X batchsize
     #      %
     
     tmp <- size(input_batch)  # basically dim
     numwords <- tmp[1]
     batchsize <- tmp[2]
     
     tmp <- size(weights$word_embedding)  # basically dim
     vocab_size <- tmp[1]
     numhid1 <- tmp[2]
     numhid2 <- size(weights$embed_to_hid, 2)
     
     #      %% COMPUTE STATE OF WORD EMBEDDING LAYER.
     #      % Look up the inputs word indices in the word_embedding_weights matrix.
     # each row of the word weights corresponds to a word (250 total)
     # input_batch contains 300 total words (3 * 100 batchsize)
     # each element of input_batch is a number between 1 and 249 (250?), corresponding to a word
     #embedding_layer_state2 <- matrix(as.numeric(weights$word_embedding[as.integer(input_batch), ]), nrow=numhid1 * numwords)
     #embedding_layer_state <- myReshape(weights$word_embedding[as.integer(input_batch),], nrows=numhid1 * numwords)    
     #embedding_layer_state <- reshape(t(word_embedding_weights[reshape(input_batch, 1, []), ]), numhid1 * numwords, [])
     
     # [] is allowed in reshape: one dimension remains unspecified and Octave will determine it automatically
     tmp <- t(weights$word_embedding[reshape(input_batch, 1, length(input_batch)), ])
     embedding_layer_state <- reshape(tmp, numhid1 * numwords, length(tmp) / (numhid1 * numwords))
     
     #embedding_layer_state2 <- matlab::reshape(as.matrix(weights$word_embedding[as.numeric(input_batch), ]), numhid1*numwords, 100)
     
     #      %% COMPUTE STATE OF HIDDEN LAYER.
     #      % Compute inputs to hidden units.
     # crossprod = t(x) %*% y
     #inputs_to_hidden_units = myCrossProd(weights$embed_to_hid, embedding_layer_state) + fn(weights$hid_bias, 1, batchsize)     
     inputs_to_hidden_units <- t(weights$embed_to_hid) %*% embedding_layer_state + repmat(weights$hid_bias, 1, batchsize)
     
     
     #      benchmark(
     #           tmp1 <- repmat(weights$hid_bias, 1, batchsize),
     #           tmp2 <- myRepMat4(weights$hid_bias, 1, batchsize),
     #           replications=10
     #           )
     
     
     #      % Apply logistic activation function.
     #      % FILL IN CODE. Replace the line below by one of the options.
     #       hidden_layer_state = zeros(numhid2, batchsize)
     #      % Options
     #      % (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
     #      % (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
     hidden_layer_state = 1 / (1 + exp(-inputs_to_hidden_units))
     #      % (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));
     
     #      %% COMPUTE STATE OF OUTPUT LAYER.
     #      % Compute inputs to softmax.
     #      % FILL IN CODE. Replace the line below by one of the options.
     #       inputs_to_softmax = zeros(vocab_size, batchsize)
     #      % Options
     #inputs_to_softmax = myCrossProd(weights$hid_to_output, hidden_layer_state) +  fn(weights$output_bias, 1, batchsize)
     inputs_to_softmax <- t(weights$hid_to_output) %*% hidden_layer_state + repmat(weights$output_bias, 1, batchsize)
     
     #      % (b) inputs_to_softmax = t(hid_to_output_weights) %*% hidden_layer_state +  repmat(output_bias, batchsize, 1);
     #      % (c) inputs_to_softmax = hidden_layer_state %*% t(hid_to_output_weights) +  repmat(output_bias, 1, batchsize);
     #      % (d) inputs_to_softmax = hid_to_output_weights %*% hidden_layer_state +  repmat(output_bias, batchsize, 1);
     
     #      % Subtract maximum. 
     #      % Remember that adding or subtracting the same constant from each input to a
     #      % softmax unit does not affect the outputs. Here we are subtracting maximum to
     #      % make all inputs <= 0. This prevents overflows when computing their
     #      % exponents.
     # max in matlab returns max from each column by default
     #      benchmark(
     #      tmp <- apply(inputs_to_softmax, 2, max),
     #      tmp2 <- colMaxs(inputs_to_softmax),
     #      replications=10)
     #tmp <- apply(inputs_to_softmax, 2, max)
     #inputs_to_softmax = inputs_to_softmax - fn(tmp, vocab_size, 1)
     inputs_to_softmax <- inputs_to_softmax - repmat(colMaxs(inputs_to_softmax), vocab_size, 1)
     
     #      % Compute exp.
     output_layer_state = exp(inputs_to_softmax)
     
     #      % Normalize to get probability distribution.
     #output_layer_state = output_layer_state / fn(colSums(output_layer_state), vocab_size, 1)
     output_layer_state <- output_layer_state / repmat(matlab::sum(output_layer_state), vocab_size, 1)
     
     return(list(embedding_layer_state=embedding_layer_state, 
                 hidden_layer_state=hidden_layer_state, 
                 output_layer_state=output_layer_state))
}