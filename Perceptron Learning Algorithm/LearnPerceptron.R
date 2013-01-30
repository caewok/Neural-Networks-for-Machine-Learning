# Learns the weights of a perceptron and displays the results.
source("Perceptron Learning Algorithm/PlotPerceptron.R")

learn_perceptron <- function(neg_examples_nobias,pos_examples_nobias,w_init,w_gen_feas) {
#      %% 
#      % Learns the weights of a perceptron for a 2-dimensional dataset and plots
#      % the perceptron at each iteration where an iteration is defined as one
#      % full pass through the data. If a generously feasible weight vector
#      % is provided then the visualization will also show the distance
#      % of the learned weight vectors to the generously feasible weight vector.
#      % Required Inputs:
#      %   neg_examples_nobias - The num_neg_examples x 2 matrix for the examples with target 0.
#      %       num_neg_examples is the number of examples for the negative class.
#      %   pos_examples_nobias - The num_pos_examples x 2 matrix for the examples with target 1.
#      %       num_pos_examples is the number of examples for the positive class.
#      %   w_init - A 3-dimensional initial weight vector. The last element is the bias.
#      %   w_gen_feas - A generously feasible weight vector.
#      % Returns:
#      %   w - The learned weight vector.
#      %%
     
     # Bookkeeping
     num_neg_examples = nrow(neg_examples_nobias)  # assumes 2-dimensions
     num_pos_examples = nrow(pos_examples_nobias)  # assumes 2 dimensions
     num_err_history = NULL
     w_dist_history = NULL

     # Here we add a column of ones to the examples in order to allow us to learn bias parameters
     neg_examples <- cbind(neg_examples_nobias, 1)
     pos_examples <- cbind(pos_examples_nobias, 1)
     
     # If weight vectors have not been provided, initialize them appropriately.
     if(missing(w_init) | length(w_init) == 0) {
          w <- rnorm(3)
     } else w <- w_init
     
     if(missing(w_gen_feas)) w_gen_feas <- NULL
     
     # Find the data points that the perceptron has incorrectly classified and record the number of errors it makes.
     iter <- 0
     res <- eval_perceptron(neg_examples, pos_examples, w)
     mistakes0 <- res$mistakes0
     mistakes1 <- res$mistakes1
     num_errs <- length(mistakes0) + length(mistakes1)
     num_err_history <- c(num_err_history, num_errs)
     
     printf("Number of errors in iteration %d:\t%d\n", iter, num_errs)
     printf("weights:\t%f\n", w)
     # or use err_message <- sprintf(); message(err_message)
     
     # If a generously feasible weight vector exists, record the distance to it from the initial weight vector.
     if(length(w_gen_feas) != 0) w_dist_history <- c(w_dist_history, norm(w - w_gen_feas, type="2"))
     
     plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)
     key <- readline("<Press 'enter' to continue, 'q' to quit.>")
     if(key == 'q') return(w)
     

     
     # Iterate until the perceptron has correctly classified all points.
     while (num_errs > 0) {
          iter <- iter + 1
          
          # Update the weights of the perceptron.
          w <- update_weights(neg_examples, pos_examples, w)
          
          # If a generously feasible weight vector exists, record the distance to it from the initial weight vector.
          if(length(w_gen_feas) != 0) w_dist_history <- c(w_dist_history, norm(w - w_gen_feas, type="2"))
          
          # Find the data points that the perceptron has incorrectly classified and record the number of errors it makes.
          res <- eval_perceptron(neg_examples, pos_examples, w)
          mistakes0 <- res$mistakes0
          mistakes1 <- res$mistakes1
          num_errs <- length(mistakes0) + length(mistakes1)
          num_err_history <- c(num_err_history, num_errs)
          
          printf("Number of errors in iteration %d:\t%d\n", iter, num_errs)
          printf("weights:\t%f\n", w)
          # or use err_message <- sprintf(); message(err_message)
          
          plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history)
          key <- readline("<Press 'enter' to continue, 'q' to quit.>")
          if(key == 'q') return(w)    
     }
     return(list(w=w, w_dist_history=w_dist_history))
}


eval_perceptron <- function(neg_examples, pos_examples, w) {
#      %% 
#      % Evaluates the perceptron using a given weight vector. Here, evaluation
#      % refers to finding the data points that the perceptron incorrectly classifies.
#      % Inputs:
#      %   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
#      %       num_neg_examples is the number of examples for the negative class.
#      %   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
#      %       num_pos_examples is the number of examples for the positive class.
#      %   w - A 3-dimensional weight vector, the last element is the bias.
#      % Returns:
#      %   mistakes0 - A vector containing the indices of the negative examples that have been
#      %       incorrectly classified as positive.
#      %   mistakes0 - A vector containing the indices of the positive examples that have been
#      %       incorrectly classified as negative.
#      %%
     
     num_neg_examples = nrow(neg_examples)  # assumes 2-dimensions
     num_pos_examples = nrow(pos_examples)  # assumes 2 dimensions
     
     mistakes0 <- NULL
     mistakes1 <- NULL
     
     for(i in 1:num_neg_examples) {
          x <- neg_examples[i,]
          activation <- x %*% w
          if(activation >= 0) mistakes0 <- c(mistakes0, i)  
     }
     
     for(i in 1:num_pos_examples) {
          x <- pos_examples[i,]
          activation <- x %*% w
          if(activation < 0) mistakes1 <- c(mistakes1, i) 
     }
     
     return(list(mistakes0=mistakes0, mistakes1=mistakes1))
}

# %WRITE THE CODE TO COMPLETE THIS FUNCTION
update_weights <- function(neg_examples, pos_examples, w_current) {
# %% 
#      % Updates the weights of the perceptron for incorrectly classified points
# % using the perceptron update algorithm. This function makes one sweep
# % over the dataset.
# % Inputs:
#      %   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
# %       num_neg_examples is the number of examples for the negative class.
# %   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
# %       num_pos_examples is the number of examples for the positive class.
# %   w_current - A 3-dimensional weight vector, the last element is the bias.
# % Returns:
#      %   w - The weight vector after one pass through the dataset using the perceptron
# %       learning rule.
# %%
     w <- w_current
     
     num_neg_examples = nrow(neg_examples)  # assumes 2-dimensions
     num_pos_examples = nrow(pos_examples)  # assumes 2 dimensions
     
     for(i in 1:num_neg_examples) {
          this_case <- neg_examples[i,]
          activation <- this_case %*% w
          if(activation >= 0) {
               # YOUR CODE HERE
               # output unit incorrectly outputted a 1; subtract input vector from the weight vector
               w <- w - this_case
          }
     }
     
     for(i in 1:num_pos_examples) {
          this_case <- pos_examples[i,]
          activation <- this_case %*% w
          if(activation < 0) {
               # YOUR CODE HERE
               # output unit incorrectly outputted a 0; add input vector to the weight vector
               w <- w + this_case  
          }
     }
     
     return(w)
            
}