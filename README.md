# 56042025_Churning_Customers

# Customer Churn Prediction Model

## Introduction

Customer churn is a major problem and one of the most important concerns for large companies. Due to the direct effect on the revenues of the companies, especially in the telecom field, companies are seeking to develop means to predict potential customer to churn. Therefore, finding factors that increase customer churn is important to take necessary actions to reduce this churn. The main contribution of this work is to develop a churn prediction model which assists telecom operators to predict customers who are most likely subject to churn. The model will be developed using Python and its libraries.

## Dataset

The dataset used in this project is provided in the files: [Dataset](CustomerChurn_dataset.csv).  The “Churn” column is our target.

## Data Preprocessing

The dataset is preprocessed using the following steps:

1.	Remove the customerID column as it is not needed for the prediction.
2.	Impute the rows with missing values in the appropriate columns.
3.	Convert the categorical columns to numerical columns using label encoding.
4. Transform the TotalCharges column to numerical column using pd.to_numeric() function.
5.	Scale the numerical columns using StandardScaler.

## Exploratory Data Analysis

The dataset is explored using the following steps:

1.	Plot the distribution of the target variable.
2.	Plot the distribution of the numerical columns.
3.	Plot the distribution of the categorical columns.

## Feature Selection

To enhance the model's performance and efficiency, a feature selection technique called SelectKBest was used. This helped in choosing the most relevant features for training the model. The top 10 features were selected using the `SelectKBest` function. The selected features are the features presented on the form the user fills.

## Model Building

The dataset is split into training and testing sets using train_test_split() function. The training set is used to train the model and the testing set is used to evaluate the model. A MulitLayer Perceptron (MLP) model is built using the training set. The model is evaluated using the testing set. The model is then saved using joblib.dump() function. The training was done using the Keras Functional API. The training process involved cross-validation and hyperparameter tuning using GridSearchCV and RandomSearchCV to find the optimal set of parameters.

## Ensemble Modeling
An ensemble model was created by combining three different models:

1. Unoptimized Model: This model was not fine-tuned, but its predictions were given higher weight in the ensemble because its AUC and accuracy scores were high.
2. Optimized Model 1: A version of the model with optimized hyperparameters, given equal weight as the unoptimized model in the ensemble.
3. Optimized Model 2: Another optimized version of the model, also given the lowest weight in the ensemble becuase it tested 1.5% lower.

The ensemble prediction is a weighted average of the individual models' predictions.

## Model Evaluation

The model is evaluated using the following metrics:

1.	Accuracy Score
2.	AUC Score

## Model Deployment

The model is deployed using Flask (localhost).

## Model Confidence
To improve the model's confidence estimates, a specific approach was taken. The confidence level for each prediction is calculated by penalizing values closer to 0.5 more severely. This helps the model express higher confidence when it is more certain about its predictions.

# Usage
## Training
To train the model, execute the `56042025_Churning_Customers.ipynb` Colab Notebook. This notebook contains the step-by-step process, allowing you to understand and replicate the training procedure.

## Model Serving
To serve the model and make predictions, run the `predict.py` Flask script. This script encapsulates the model and provides a simple web interface for making predictions.

## Model Ensemble Weights
### In the ensemble model:

The Unoptimized Model is given a weight of 2.
Optimized Model 1 and Optimized Model 2 are given weights of 2 and 1 respectively.

Adjusting these weights can have an impact on the ensemble's overall performance.

# Link to Demo Video

# https://youtu.be/sQ0du2TSKHo 
