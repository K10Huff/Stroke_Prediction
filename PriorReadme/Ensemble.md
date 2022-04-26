# Ensemble Learning

## Overview

For the ensemble section we decided to compare 2 models to understand which model would give the best results and for what reasons.

The models that were compared were:

- Balanced Random Forest Classifier

- Easy Ensemble AdaBoost Classifier

In preparation to compare the models there was an additional preprocessing step that was completed. We completed a one-hot encoding on the categorical data, as well as a standard scaler on the numerical data. The standard scaler changes would allow us to have a more consistently distributed dataset from which to train our model, and the one-hot encoding allows the model to understand the categorical values and utilize them in training.

## Ensemble Classifiers

Ensemble classifiers are methods that use multiple learning algorithms to obtain better predictive analysis.

### Balanced Random Forest Classifier

*Balanced random forest classifier is a method that randomly undersamples each bootstrap sample to balance it*

```
Balanced Random Forest Classifier

Balanced Accuracy Score: 0.7757064364207221

                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  852               324
Actual Stroke                       9                43

                   pre       rec       spe        f1       geo       iba       sup

          0       0.99      0.72      0.83      0.84      0.77      0.59      1176
          1       0.12      0.83      0.72      0.21      0.77      0.61        52

avg / total       0.95      0.73      0.82      0.81      0.77      0.59      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].

### Easy Ensemble AdaBoost Classifier

*Easy Ensemble AdaBoost Classifier is an ensemble of AdaBoost trained on different balanced bootstrap samples which are achieved by under-sampling*

```
Easy Ensemble AdaBoost Classifier

Balanced Accuracy Score: 0.7162480376766092

                  Predicted No Stroke  Predicted Stroke
Actual No Stroke                  780               396
Actual Stroke                      12                40

                   pre       rec       spe        f1       geo       iba       sup

          0       0.98      0.66      0.77      0.79      0.71      0.50      1176
          1       0.09      0.77      0.66      0.16      0.71      0.52        52

avg / total       0.95      0.67      0.76      0.77      0.71      0.51      1228
```

For the results we see that overall precision is quite good, however the precision of predicted stoke is very low. This shows a high level of type 1 error [false positive].

## Conclusions

Overall we see that the ensemble methods fare quite similarly in their results. They all seem very susceptible to type 1 error. Albeit in this scenario, type 1 error, or false positive is far more ideal than false negatives, which could lead to strokes that would not be predicted. The random forest classifier has far better recall however resulting in a higher f1 score. The stark difference of these two models can be traced to their inability to consistently discern when an actual stroke is occurring.
