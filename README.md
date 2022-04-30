# Stroke Prediction
## Topic Selected
 The goal of this analysis is to put together a stroke prediction model using various health and lifestyle metrics. It is important to understand and assess the risk of suffering from a stroke and the purpose of this investigation is to apply different machine learning models to empower people with information.

## Dataset
The dataset was sourced from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset. The dataset includes eleven different features regarding health and lifestyle. The goal is to determine if there is a way to determine a patient’s risk for stroke based on the eleven features.

### Initial Analysis of the Data

![ Tableau 1 ](github link)
![ Tableau 2 ](github link)

### Questions to be Addressed: 
- Which learning model work best with our dataset
- Which features influence the model the most?
- What metrics best determine stroke risk.

## Results/ Visualization 
![ Result LM 1 ](github link)
![ Result LM 2 ](github link)

# Ensemble Learning

### Overview

For the ensemble section we decided to compare 2 models to understand which model would give the best results and for what reasons.

The models that were compared were:

- Balanced Random Forest Classifier

- Easy Ensemble AdaBoost Classifier

In preparation to compare the models there was an additional preprocessing step that was completed. We completed a one-hot encoding on the categorical data, as well as a standard scaler on the numerical data. The standard scaler changes would allow us to have a more consistently distributed dataset from which to train our model, and the one-hot encoding allows the model to understand the categorical values and utilize them in training.

### Ensemble Classifiers

Ensemble classifiers are methods that use multiple learning algorithms to obtain better predictive analysis.

#### Balanced Random Forest Classifier

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

#### Easy Ensemble AdaBoost Classifier

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

### Feature Importance

We wanted to take a look at the specific inputs that the model was using to make its classifying decision. These were the percentage weight it was putting on each of the inputted features. 


<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>

<table class="tablelines">
  <thead>
    <tr>
      <th>Feature</th>
      <th>Weighted Percentage</th>
     </tr>
  </thead>
  <tbody>
    <tr>
      <td>Age</td>
      <td>0.33983</td>
     </tr>
    <tr>
      <td>Avg Glucose Level</td>
      <td>0.17349</td>
    </tr>
    <tr>
      <td>BMI</td>
      <td>0.14972</td>
    </tr>
    <tr>
      <td>Hypertension</td>
      <td>0.042611</td>
    </tr>
    <tr>
      <td>Work Type: Self-employed</td>
      <td>0.026672</td>
    </tr>
    <tr>
      <td>Heart Disease</td>
      <td>0.026672</td>
    </tr>
    <tr>
      <td>Ever Married: No</td>
      <td>0.024329</td>
    </tr>   
    <tr>
      <td>Ever Married: Yes</td>
      <td> 0.023116</td>
    </tr>
    <tr>
      <td>Smoking Status: Unknown</td>
      <td>0.020450</td>
    </tr>
    <tr>
      <td>Work Type: Private</td>
      <td>0.020072</td>
    </tr>
    <tr>
      <td>Smoking Status: Never Smoked </td>
      <td>0.019952</td>
    </tr>
    <tr>
      <td>Smoking Status: Formerly Smoked</td>
      <td>0.019684</td>
    </tr> 
    <tr>
      <td>Work Type: Children</td>
      <td>0.018427</td>
    </tr>
    <tr>
      <td>Residence Type: Urban</td>
      <td>0.017786</td>
    </tr>
    <tr>
      <td>Gender: Male</td>
      <td>0.016488</td>
    </tr>
    <tr>
      <td>Residence Type: Rural</td>
      <td>0.016397</td>
    </tr>
    <tr>
      <td>Smoking Status: Smokes</td>
      <td>0.015680</td>
    </tr>      
    <tr>
      <td>Gender: Female</td>
      <td>0.014850</td>
    </tr>
    <tr>
      <td>Work Type: Goverment Job</td>
      <td>0.013635</td>
    </tr>
    <tr>
      <td>Work Type: Never Worked</td>
      <td>0.00014072</td>
    </tr>
    <tr>
      <td>Gender: Other</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>




#### Conclusions

Overall we see that the ensemble methods fare quite similarly in their results. They all seem very susceptible to type 1 error. Albeit in this scenario, type 1 error, or false positive is far more ideal than false negatives, which could lead to strokes that would not be predicted. The random forest classifier has far better recall however resulting in a higher f1 score. The stark difference of these two models can be traced to their inability to consistently discern when an actual stroke is occurring.