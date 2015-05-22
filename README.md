## Python Code for BBVI - 4D-Gaussian Example
This is the Python implemetnation Rajesh et. al's Black Box Variational Inference for a simple 4-d Gaussian model. 
The code is here more as a proof of concept that this works and contains the AdaGrad extension along with the MCCV
needed to control the variance of the estimator. It should be in theory easy to modify this to work with a different model
by replacing the gradient of the variational distribution (the Gaussian) with your distribution of choice as well
as the log joint probability of the model. 

## Package Requirements
Pandas
Numpy
Matplotlib
Seaborn
Scipy

## Running this Code
To run simply type:
`python bbvi_simple.py`

## Plots Generated
bbvi_gaussian_samples.png - A scatter plot of the first two dimensions for the prior / likelihood / posterior. 
bbvi_gaussian.png - A facet plot that shows the sample paths for the various implementations of BBVI where the learning rate 
can be either Robbins-Monroe or AdaGrad and whether Markov Control Variates are used. The diagonal plots represent a histogram of the samples collected from every iteration of BBVI where the yellow represents the true value (made slightly larger for visibility).

![alt tag](https://github.com/daeilkim/bbvi/blob/master/bbvi_gaussian_samples.png)
![alt tag](https://github.com/daeilkim/bbvi/blob/master/bbvi_gaussian.png)
