# Stroke Prediction

## Topic Selected

 The goal of this analysis is to put together a stroke prediction model using various health and lifestyle metrics. It is important to understand and assess the risk of suffering from a stroke and the purpose of this investigation is to apply different machine learning models to empower people with information.

### Questions to be Addressed:

- Which learning model work best with our dataset
- Which features influence the model the most?

### Dataset

The dataset was sourced from https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset. The dataset includes eleven different features regarding health and lifestyle. The goal is to determine if there is a way to determine a patient’s risk for stroke based on the eleven features.

## Initial Analysis of the Data

Overview of the Features
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/Feature_Matrix.png" alt="Overview of the Features">

Subjects in the Dataset Who Have Suffered A Stroke
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/Target_variable.png" alt="Target Variable">

Health Metrics
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/Hyper-Heart-BMI_Vs_Stroke_Risk.png" alt="Hyper_Heart_BMI">

Heatmap of Smoking Status Brokendown by Age
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/HEATMAP-Smoking_Vs_Stroke_Risk.png" alt="Heatmap_Smoking">

Breakdown of Residence Type by Gender and Smoking Status
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/residence_smoking_vs.storke_1.png" alt="Residence_Smoking_Gender">

Risk vs Marital Status, Job Type, and Age
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/Dashboard%231.png" alt="Dashboard">

Heatmap of Married and Residence Type Broken Down by Gender
<img src="https://github.com/K10Huff/Team_01_Project/blob/6701451306c5c67d7594a3d7e6c085f497d190db/resources/images/HEATMAP-Married_Gender_Residence_Vs_Stroke_Risk.png" alt="Married_Heatmap">

## Results & Visualizations

### Ensemble Learning

#### Overview

For the ensemble section we decided to compare 2 models to understand which model would give the best results and for what reasons.

The models that were compared were:

- Balanced Random Forest Classifier

- Easy Ensemble AdaBoost Classifier

In preparation to compare the models there was an additional preprocessing step that was completed. We completed a one-hot encoding on the categorical data, as well as a standard scaler on the numerical data. The standard scaler changes would allow us to have a more consistently distributed dataset from which to train our model, and the one-hot encoding allows the model to understand the categorical values and utilize them in training.

#### Ensemble Classifiers

Ensemble classifiers are methods that use multiple learning algorithms to obtain better predictive analysis.

##### Balanced Random Forest Classifier

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

##### Easy Ensemble AdaBoost Classifier

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

#### Ensemble Classifier Conclusions

Overall we see that the ensemble methods fare quite similarly in their results. They all seem very susceptible to type 1 error. Albeit in this scenario, type 1 error, or false positive is far more ideal than false negatives, which could lead to strokes that would not be predicted. The random forest classifier has far better recall however resulting in a higher f1 score. The stark difference of these two models can be traced to their inability to consistently discern when an actual stroke is occurring.

### Sampling

#### Overview

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

#### Oversampling

Oversampling is a method that creates new samples (in some form) of the minority class.

##### Naïve Random

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

##### SMOTE

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

##### ADASYN

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

#### Undersampling

Undersampling is a technique that is used to help in leveling out datasets by keeping all the data in the minority class and decreasing the size of the majority class.

##### Cluster Centroids

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

#### Combination (Over/Under)

Combo uses techniques from both over and undersampling in an attempt to create a more balanced dataset.

##### SMOTEEN

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

#### Sampling Conclusions

Overall we see that the sampling methods fare quite similarly in their results. They all seem very susceptible to type 1 error. Albeit in this scenario, type 1 error, or false positive is far more ideal than false negatives, which could lead to strokes that would not be predicted. They all boast similar F1 scores, but the BAC (Balanced Accuracy Score) of the SMOTEEN and ADASYN models are higher. Given that the dataset we are using is on the smaller side, it may be more beneficial to use an oversampling technique to do the training as it would generate more points in the minority class.

### Neural Networks

#### Overview

Neural Networks are machine learning strategies modeled after the human brain. Within neural networks, computations are performed by a neuron or a perceptron and have the ability to pass the data on to other neurons or other layers within the model. The benefit to this model is that after the initial layer, the subsequent perceptrons are working with weighted data instead of raw data.  
In these models, dense refers to the parameter that all perceptrons are interconnected and ReLU and Sigmoid are activation functions that weight the data in different ways. We were interested in a classification/binary decision and so only one output neuron with a sigmoid activation function was necessary to produce a probability output. During compiling the adam optimizer was used to help the model overcome weaker features and the binary_crossentropy loss function was chosen since it was specifically designed to evaluate a binary classification model.

#### Single Perceptron

*The simplest of the neural networks, it contains a single perceptron and only two layers. The input layer passes the data to the perceptron and the perceptron performs a set of calculations and passes the weighted data to an activation function for the output layer.*

#### Multi Perceptron, Single Layer

*In this step the model is elevated to an Artificial Neural Network (ANN) with the addition of other perceptrons.*

#### Deep Learning

*A network of multiple layers named hidden layers, holding multiple perceptions. There are different types of interconnectedness including Convolution Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory Networks (LSTM). Here the relatively simple dense, feed-forward network was used.*

#### Neural Network Results

<img src="https://github.com/K10Huff/Team_01_Project/blob/d5653208794722e7aebb4947f6dbd4897d55692e/resources/images/history_plots.png" alt="history_plots">
<img src="https://github.com/K10Huff/Team_01_Project/blob/d5653208794722e7aebb4947f6dbd4897d55692e/resources/images/confusion_matrices.png" alt="confusion_matrices">
<img src="https://github.com/K10Huff/Team_01_Project/blob/d5653208794722e7aebb4947f6dbd4897d55692e/resources/images/total_acm_df.png" alt="total_acm_df">

#### Neural Network Conclusion

Here we can see that as the neural network becomes more complex, it becomes more susceptible to overfitting; both the precision and the balanced accuracy scored drop significantly. The simplest, single perceptron model yields the best results. 

## Feature Importance

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

## Project Conclusion
