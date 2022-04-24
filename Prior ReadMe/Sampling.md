# Sampling

## Overview

For the sampling section we decided to compare models to understand which model would give the best results and for what reasons. 

The models that were compared were:

- Oversampling
  
  - Naïve Random
  
  - SMOTE
  
  - ADASYN

- Undersampling
  
  - Cluster Centroids

- Comination (Over/Under)
  
  - SMOTEEN

In preparation to compare the models there was an additional preprocessing step that was completed. We completed a one-hot encoding on the categorical data, as well as a standard scaler on the numerical data. The standard scaler changes would allow us to have a more consistently distributed dataset from which to train our model, and the one-hot encoding allows the model to understand the categorical values and utilize them in training. 

## Oversampling

Oversampling is a method that creates new samples (in some form) of the minority class.

### Naïve Random

*Naive Random sampling is a method that duplicates examples from the minority class.*

```
Naive Random Oversampling
Balanced Accuracy Score: 0.7540881737310309
                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  869               307
Actual Stroke                      12                40
                   pre       rec       spe        f1       geo       iba       sup

          0       0.99      0.74      0.77      0.84      0.75      0.57      1176
          1       0.12      0.77      0.74      0.20      0.75      0.57        52

avg / total       0.95      0.74      0.77      0.82      0.75      0.57      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].

### SMOTE

*Smote is a method that generates synthetic samples from the minority class.*

```
SMOTE Oversampling
Balanced Accuracy Score: 0.7446363160648874

                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  892               284
Actual Stroke                      14                38

                   pre       rec       spe        f1       geo       iba       sup

          0       0.98      0.76      0.73      0.86      0.74      0.56      1176
          1       0.12      0.73      0.76      0.20      0.74      0.55        52

avg / total       0.95      0.76      0.73      0.83      0.74      0.56      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].

### ADASYN

*ADASYN is a method that uses a weighted distribution for different minority class examples based on their level of difficulty in learning.*

```
ADASYN Oversampling
Balanced Accuracy Score: 0.7724686028257457

                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  867               309
Actual Stroke                      10                42

                   pre       rec       spe        f1       geo       iba       sup

          0       0.99      0.74      0.81      0.84      0.77      0.59      1176
          1       0.12      0.81      0.74      0.21      0.77      0.60        52

avg / total       0.95      0.74      0.80      0.82      0.77      0.59      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].

## Undersampling

Undersampling is a technique that is used to help in leveling out datasets by keeping all the data in the minority class and decreasing the size of the majority class.

### Cluster Centroids

*Cluster Centroids is a method that under samples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm.*

```
Cluster Centroids Undersampling
Balanced Accuracy Score: 0.7243916797488226

                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  867               309
Actual Stroke                      15                37

                   pre       rec       spe        f1       geo       iba       sup

          0       0.98      0.74      0.71      0.84      0.72      0.53      1176
          1       0.11      0.71      0.74      0.19      0.72      0.52        52

avg / total       0.95      0.74      0.71      0.81      0.72      0.53      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].

## Combination (Over/Under)

Combo uses techniques from both over and undersampling in an attempt to create a more balanced dataset.

### SMOTEEN

*SMOTEEN is a method that combines the SMOTE ability to generate synthetic examples for the minority class and ENN ability to delete some observations from both minority and majority classes.*

```
SMOTEENN
Balanced Accuracy Score: 0.763376504447933

                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  823               353
Actual Stroke                       9                43

                   pre       rec       spe        f1       geo       iba       sup

          0       0.99      0.70      0.83      0.82      0.76      0.57      1176
          1       0.11      0.83      0.70      0.19      0.76      0.59        52

avg / total       0.95      0.71      0.82      0.79      0.76      0.57      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].



## Conclusions

Overall we see that the sampling methods fare quite similarly in their results. They all seem very susceptible to type 1 error. Albeit in this scenario, type 1 error, or false positive is far more ideal than false negatives, which could lead to strokes that would not be predicted. They all boast similar F1 scores, but the BAC (Balanced Accuracy Score) of the SMOTEEN and ADASYN models are higher. Given that the dataset we are using is on the smaller side, it may be more beneficial to use an oversampling technique to do the training as it would generate more points in the minority class.


