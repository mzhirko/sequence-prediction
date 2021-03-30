# Sequence Prediction
## Objective
A numerical sequence is given: (x0, ..., xq), where xi = f(t+i*h). The implemented model, after training on a sample of L = q-p images (xk, ...,xk+p), where p < q and k = 0, ... q-p-1, whose reference values are xk+p+1, should provide prediction of the p+i-th value (i> 1), for an arbitrary sequence of p+1 values. The model must provide scaling of the values of a given sequence for any range, if the activation function used so requires. 

## The interface of the model provides
* the ability to specify p explicitly; 
* possibility to specify the maximum allowable root-mean-square error of sampling; 
* possibility to specify maximum admissible step of training; 
* the ability to specify the maximum allowable number of training iterations; 
* the possibility of specifying on/off the mode of obligatory zeroing, separately for the first and for every following iteration, of the context neurons both at training as well as during prediction; 
* ability to automatically predict n values from p+2 to p+1+n; 
* the possibility to display the values of weights and thresholds for each layer for the current iteration;

## Input
The sequence X of the k=q+1 length (where q > 0), by which this network will be trained.
