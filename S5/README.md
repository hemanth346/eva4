## Model 1

### Target :

1. Get the setup right
    1. See/know the data 
> (skipping for mnist)
        1. Set basic transformation(to_tensor)
        1. Decide RF for edges/gradients
        1. Get Mean and standard deviation values of the dataset
        1. Decide transforms to use
    1. Set Transformations compose
    1. Set Data Loader
    1. Set Basic working code
    1. Set Basic Training and Test loop

1. Set the architecture
    1. Transformational block after RF of 5
    1. Get Conv layer before GAP to not less than 5px to retain info
    1. Use a FC/pointwise after GAP (allows usage of more features for prediction)
    1. Use less than 9.5k Param, Fully-Conv network
    1. Achieve round 99% train and test acc
    1. No BN, No dropout

### Results :
    
1. Parameters: 8,570
1. Best Train Accuracy: 98.72
1. Best Test Accuracy: 98.97 

### Analysis
1. Model is under-fitting slightly. 
1. Could be due to below reasons
    - Since model is not complex enough it may not able to genaralize properly
    - Model architecture is the culprit, could be due to maxpool is used early on due to param  restriction and o/p size 


## Model 2

### Target :

1. Modify architecture 
    1. Use than 9.5k param, FCN
    1. only 1 max pool
    1. No BN, Dropout 
    1. Achieve round 99% train and test acc
2. No under-fitting

### Results :
1.  Parameters: 9,290
1. Best Train Accuracy: 98.95
1. Best Test Accuracy: 98.85 

### Analysis
1. Achieved target
1. Good Model. No under-fitting
1. Is capable if pushed further (more epochs and/or augmented data)


## Model 3

### Target :

1. Add Batch-norm to increase model efficiency

### Results :
1.  Parameters: 9,430
1. Best Train Accuracy: 99.55
1. Best Test Accuracy: 99.41 

### Analysis
1. Achieved target accuracy but only for 1 epoch
1. Additional 140 param added due to BatchNorm
1. We have started to see very slight over-fitting now, but we can push the model a little further with out regularization.
1. From the plot - loss jumping around after 4th epoch, we can use LR manager to try and find global minima smoothly


## Model 4

### Target :

1. Add LR manager

### Results :
1.  Parameters: 9,430
1. Best Train Accuracy: 99.69
1. Best Test Accuracy: 99.44 

### Analysis
1. Achieved target accuracy for mulitple epochs
1. Many epochs with accuracy above 99.38
1. Over-fitting is having effect on the model now. Add regularization next
1. Plots gone crazy.! On googling, found that these plots are similar to cosine annealing learning rate (not sure what that is, yet).


## Model 5

### Target :

1. Add regularization, Dropout

### Results :
1. Parameters: 9,430
1. Best Train Accuracy: 99.47
1. Best Test Accuracy: 99.46 

### Analysis
1. Achieved target accuracy more consistently for mulitple epochs
1. Good model, can be pushed further if required.
