"""
2. EM algorithm 
 
INPUT: 
MNIST training data and label sets. 

 FUNCTION: 
 a. Binning the gray level value into two bins. Treating all pixels as random variables following Bernoulli distributions. Note that each pixel follows a different Binomial distribution independent to others. 

 b. Use EM algorithm to cluster each image into ten groups. You should come up with a reasonable rule to determine convergence. (a simple run out of the loop should be used as the ultimatum) 

 OUTPUT: 
For each digit, output a confusion matrix and the sensitivity and specificity of the clustering applied to the training data.
"""