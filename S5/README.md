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
    1. No BN, dropout

### Results :
    
1.  Parameters: 8,570
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
1. Achived target
1. Good Model. No under-fitting
1. Is capable if pushed further (more epochs and/or augmented data)
