# %% Plots information about a perceptron classifier on a 2-dimensional dataset.
plot_perceptron <- function(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, w, w_dist_history) {
# %%
# % The top-left plot shows the dataset and the classification boundary given by
# % the weights of the perceptron. The negative examples are shown as circles
# % while the positive examples are shown as squares. If an example is colored
# % green then it means that the example has been correctly classified by the
# % provided weights. If it is colored red then it has been incorrectly classified.
# % The top-right plot shows the number of mistakes the perceptron algorithm has
# % made in each iteration so far.
# % The bottom-left plot shows the distance to some generously feasible weight
# % vector if one has been provided (note, there can be an infinite number of these).
# % Points that the classifier has made a mistake on are shown in red,
# % while points that are correctly classified are shown in green.
# % The goal is for all of the points to be green (if it is possible to do so).
# % Inputs:
# %   neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
# %       num_neg_examples is the number of examples for the negative class.
# %   pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
# %       num_pos_examples is the number of examples for the positive class.
# %   mistakes0 - A vector containing the indices of the datapoints from class 0 incorrectly
# %       classified by the perceptron. This is a subset of neg_examples.
# %   mistakes1 - A vector containing the indices of the datapoints from class 1 incorrectly
# %       classified by the perceptron. This is a subset of pos_examples.
# %   num_err_history - A vector containing the number of mistakes for each
# %       iteration of learning so far.
# %   w - A 3-dimensional vector corresponding to the current weights of the
# %       perceptron. The last element is the bias.
# %   w_dist_history - A vector containing the L2-distance to a generously
# %       feasible weight vector for each iteration of learning so far.
# %       Empty if one has not been provided.
# %%
     # set up the plot
     par(mfrow=c(2,2))
     
     # Plot 1: classification boundary
     x_lim <- c(-1, 1)
     y_lim <- c(-1, 1)
     plot(NULL, NULL, xlim=x_lim, ylim=y_lim, xlab="Column1", ylab="Column2")
     title("Classifier")
     
     neg_correct_ind <- setdiff(1:nrow(neg_examples), mistakes0)
     pos_correct_ind <- setdiff(1:nrow(pos_examples), mistakes1)
     
     # negative correct examples are green circles
     if(length(neg_correct_ind) > 0) points(neg_examples[neg_correct_ind, 1], neg_examples[neg_correct_ind, 2], type="p", col="green", pch=1, cex=2)
     
     # positive correct examples are green squares
     if(length(pos_correct_ind) > 0) points(pos_examples[pos_correct_ind, 1], pos_examples[pos_correct_ind, 2], type="p", col="green", pch=0, cex=2)
     
     # negative incorrect examples are red circles
     if(length(mistakes0) > 0) points(neg_examples[mistakes0, 1], neg_examples[mistakes0, 2], type="p", col="red", pch=1, cex=2)
     
     # positive incorrect examples are red squares
     if(length(mistakes1) > 0) points(pos_examples[mistakes1, 1], pos_examples[mistakes1, 2], type="p", col="red", pch=0, cex=2)
     
     # plot the decision line
     lines(c(-5,5), c((-w[length(w)]+5*w[1])/w[2],(-w[length(w)]-5*w[1])/w[2]))
   
     
     # Number of mistakes made thus far
     x_lim <- c(-1, max(15, length(num_err_history)))
     y_lim <- c(0, nrow(neg_examples) + nrow(pos_examples) + 1)
     plot(0:(length(num_err_history)-1), num_err_history, xlim=x_lim, ylim=y_lim, xlab="Iteration", ylab="Number of errors", main="Number of errors", type="l")
     
     # Distance to generously feasible weight vector
     x_lim <- c(-1, max(15, length(num_err_history)))
     y_lim <- c(0, 15)
     plot(0:(length(w_dist_history)-1), w_dist_history, xlab="Iteration", ylab="Distance", main="Distance", type="l")
     
}