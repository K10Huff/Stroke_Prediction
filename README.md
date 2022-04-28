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


|age                                    | 0.33983019949238036|
|avg_glucose_level                      | 0.17349014475396338|
|bmi                                    | 0.14971810735343785|
|hypertension                           | 0.04261018022895897|
|work_type_Self-employed                | 0.026672087712252122|
|heart_disease                          | 0.026671609408358332|
|ever_married_No                       |  0.024328916335137944|
|ever_married_Yes                      |  0.023115905937615477|
|smoking_status_Unknown                |  0.020449701213853793|
|work_type_Private                     |  0.02007207888484315|
|smoking_status_never smoked           |  0.019951802300754967|
|smoking_status_formerly smoked        |  0.019684002652911994|
|work_type_children                     | 0.018427413493115052|
|Residence_type_Urban                  |  0.017785897634571397|
|gender_Male                           |  0.016488190464610017|
|Residence_type_Rural                  |  0.01639706241458821|
|smoking_status_smokes                 |  0.015680279157306187|
|gender_Female                         |  0.014850498078374064|
|work_type_Govt_job                    |  0.013635204854215526|
|work_type_Never_worked                |  0.00014071762875119773|
|gender_Other                          |  0.0|
```

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
      <td>age</td>
      <td>33983019949238036</td>
     </tr>
    <tr>
      <td>avg_glucose_level</td>
      <td>0.17349014475396338</td>
    </tr>
    <tr>
      <td>bmi</td>
      <td>0.14971810735343785</td>
    </tr>
    <tr>
      <td>hypertension</td>
      <td>0.04261018022895897</td>
   
    </tr>
  </tbody>
</table>

<table class="tablelines">
  <tbody>
    <tr>
      <td>a</td>
      <td>b</td>
      <td>c</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>

<table class="tablelines">
  <thead>
    <tr>
      <th>Default aligned</th>
      <th style="text-align: left">Left aligned</th>
      <th style="text-align: center">Center aligned</th>
      <th style="text-align: right">Right aligned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>First body part</td>
      <td style="text-align: left">Second cell</td>
      <td style="text-align: center">Third cell</td>
      <td style="text-align: right">fourth cell</td>
    </tr>
    <tr>
      <td>Second line</td>
      <td style="text-align: left">foo</td>
      <td style="text-align: center"><strong>strong</strong></td>
      <td style="text-align: right">baz</td>
    </tr>
    <tr>
      <td>Third line</td>
      <td style="text-align: left">quux</td>
      <td style="text-align: center">baz</td>
      <td style="text-align: right">bar</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td>Second body</td>
      <td style="text-align: left"> </td>
      <td style="text-align: center"> </td>
      <td style="text-align: right"> </td>
    </tr>
    <tr>
      <td>2 line</td>
      <td style="text-align: left"> </td>
      <td style="text-align: center"> </td>
      <td style="text-align: right"> </td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td>Footer row</td>
      <td style="text-align: left"> </td>
      <td style="text-align: center"> </td>
      <td style="text-align: right"> </td>
    </tr>
  </tfoot>
</table>

<table class="tablelines">
  <tbody>
    <tr>
      <td>A simple</td>
      <td>table</td>
    </tr>
    <tr>
      <td>with multiple</td>
      <td>lines</td>
    </tr>
  </tbody>
</table>

#### Conclusions

Overall we see that the ensemble methods fare quite similarly in their results. They all seem very susceptible to type 1 error. Albeit in this scenario, type 1 error, or false positive is far more ideal than false negatives, which could lead to strokes that would not be predicted. The random forest classifier has far better recall however resulting in a higher f1 score. The stark difference of these two models can be traced to their inability to consistently discern when an actual stroke is occurring.