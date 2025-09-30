"""
Logistic Regression:
- Used for binary classification problems. It predicts the probability that an instance belongs to a certain class.

The Sigmoid Function
-  Logistic regression takes the linear combination of the features (just like in Linear Regression) and passes it through a special function called the Sigmoid (or Logistic) function

                        σ(z) = 1 / (1 + e^-z)
                

    - where z is the linear combination of the features: z = beta_0 + beta_1x_1 + beta_2x_2 + ... + beta_nx_n
    - Output between 0 and 1: 
        - This output can be directly interpreted as a probability.
            - If sigma(z) is close to 1, it means a high probability of belonging to the positive class.
            - If sigma(z) is close to 0, it means a high probability of belonging to the negative class.

"""


# Dataset: The Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn. 
# This dataset contains features computed from digitised images of breast mass and a target variable indicating whether the mass is malignant or benign.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler    # A preprocessing step that scales the features so they have a mean of 0 and a standard deviation of 1. Gives all features equal "weight" in terms of scale.
from sklearn.datasets import load_breast_cancer

print("-"*100)

### ----- 1. Data Loading and Initial Exploration -----

# Load the breast cancer dataset
cancer = load_breast_cancer(as_frame=True)
df_cancer = cancer.frame # Get the DataFrame of features
df_cancer['target'] = cancer.target # Add the target variable (0: malignant, 1: benign)

# Map target names for better readability
# Original: 0 = malignant, 1 = benign
df_cancer['target_names'] = df_cancer['target'].map(lambda x: cancer.target_names[x])

"""
When you call cancer = load_breast_cancer(as_frame=True):
- The cancer object is not just a DataFrame
- It's a special Bunch object (a dictionary-like object). 
- This Bunch object contains:
    - cancer.data: The feature data (what became cancer.frame).
    - cancer.target: The numerical target labels (what became df_cancer['target']).
    - cancer.target_names: This is an array that contains the human-readable names corresponding to the numerical target labels.
        - For the Breast Cancer dataset, cancer.target_names is typically ['malignant' 'benign'].

Without as_frame=True (default behavior): 
- load_breast_cancer() would return a Bunch object where cancer.data is a NumPy array of features and cancer.target is a NumPy array of target labels.
With as_frame=True: load_breast_cancer() still returns a Bunch object, but now:
- cancer.data is accessed via cancer.frame and is a Pandas DataFrame of features.
- cancer.target is a Pandas Series of target labels.
This makes it much easier to work with the data directly using Pandas DataFrame operations (like .head(), .info(), .drop(), etc.).        
        
```
df_cancer['target_names'] = df_cancer['target'].map(lambda x: cancer.target_names[x])
```
- df_cancer['target']: This is a Series containing the numerical target values (0s and 1s).
    - Example: [0, 1, 0, 1, 1, ...]
- .map(...): This is a Pandas Series method that applies a function to each element in the Series.
- lambda x: cancer.target_names[x]: This is a small, anonymous function.
    - When map processes a 0 from df_cancer['target'], x becomes 0. The function then returns cancer.target_names[0], which is 'malignant'.
    - When map processes a 1 from df_cancer['target'], x becomes 1. The function then returns cancer.target_names[1], which is 'benign'.
"""

print("--- Breast Cancer Dataset Loaded ---")
print(df_cancer.head())

# --- Breast Cancer Dataset Loaded ---
#    mean radius  mean texture  mean perimeter  mean area  mean smoothness  ...  worst concavity  worst concave points  worst symmetry  worst fractal dimension  target
# 0        17.99         10.38          122.80     1001.0          0.11840  ...           0.7119                0.2654          0.4601                  0.11890       0
# 1        20.57         17.77          132.90     1326.0          0.08474  ...           0.2416                0.1860          0.2750                  0.08902       0
# 2        19.69         21.25          130.00     1203.0          0.10960  ...           0.4504                0.2430          0.3613                  0.08758       0
# 3        11.42         20.38           77.58      386.1          0.14250  ...           0.6869                0.2575          0.6638                  0.17300       0
# 4        20.29         14.34          135.10     1297.0          0.10030  ...           0.4000                0.1625          0.2364                  0.07678       0

# [5 rows x 31 columns]

print("\n--- Data Info ---")
df_cancer.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 569 entries, 0 to 568
# Data columns (total 31 columns):
#  #   Column                   Non-Null Count  Dtype  
# ---  ------                   --------------  -----  
#  0   mean radius              569 non-null    float64
#  1   mean texture             569 non-null    float64
#  2   mean perimeter           569 non-null    float64
#  3   mean area                569 non-null    float64
#  4   mean smoothness          569 non-null    float64
#  5   mean compactness         569 non-null    float64
#  6   mean concavity           569 non-null    float64
#  7   mean concave points      569 non-null    float64
#  8   mean symmetry            569 non-null    float64
#  9   mean fractal dimension   569 non-null    float64
#  10  radius error             569 non-null    float64
#  11  texture error            569 non-null    float64
#  12  perimeter error          569 non-null    float64
#  13  area error               569 non-null    float64
#  14  smoothness error         569 non-null    float64
#  15  compactness error        569 non-null    float64
#  16  concavity error          569 non-null    float64
#  17  concave points error     569 non-null    float64
#  18  symmetry error           569 non-null    float64
#  19  fractal dimension error  569 non-null    float64
#  20  worst radius             569 non-null    float64
#  21  worst texture            569 non-null    float64
#  22  worst perimeter          569 non-null    float64
#  23  worst area               569 non-null    float64
#  24  worst smoothness         569 non-null    float64
#  25  worst compactness        569 non-null    float64
#  26  worst concavity          569 non-null    float64
#  27  worst concave points     569 non-null    float64
#  28  worst symmetry           569 non-null    float64
#  29  worst fractal dimension  569 non-null    float64
#  30  target                   569 non-null    int64  
# dtypes: float64(30), int64(1)
# memory usage: 137.9 KB


print("\n--- Data Description ---")
print(df_cancer.describe())

#        mean radius  mean texture  mean perimeter    mean area  mean smoothness  ...  worst concavity  worst concave points  worst symmetry  worst fractal dimension      target
# count   569.000000    569.000000      569.000000   569.000000       569.000000  ...       569.000000            569.000000      569.000000               569.000000  569.000000
# mean     14.127292     19.289649       91.969033   654.889104         0.096360  ...         0.272188              0.114606        0.290076                 0.083946    0.627417
# std       3.524049      4.301036       24.298981   351.914129         0.014064  ...         0.208624              0.065732        0.061867                 0.018061    0.483918
# min       6.981000      9.710000       43.790000   143.500000         0.052630  ...         0.000000              0.000000        0.156500                 0.055040    0.000000
# 25%      11.700000     16.170000       75.170000   420.300000         0.086370  ...         0.114500              0.064930        0.250400                 0.071460    0.000000
# 50%      13.370000     18.840000       86.240000   551.100000         0.095870  ...         0.226700              0.099930        0.282200                 0.080040    1.000000
# 75%      15.780000     21.800000      104.100000   782.700000         0.105300  ...         0.382900              0.161400        0.317900                 0.092080    1.000000
# max      28.110000     39.280000      188.500000  2501.000000         0.163400  ...         1.252000              0.291000        0.663800                 0.207500    1.000000

# [8 rows x 31 columns]

print("\n--- Target Variable Distribution ---")
# Count how many instances belong to each class (0: malignant, 1: benign)
print(df_cancer['target'].value_counts())
print(f"Maligant (0): {df_cancer['target'].value_counts()[0]} instances")
print(f"Benign (1): {df_cancer['target'].value_counts()[1]} instances")

# target
# 1    357
# 0    212
# Name: count, dtype: int64
# Maligant (0): 212 instances
# Benign (1): 357 instances


print("-"*100)


### ----- 2. Data Preparation -----

# Define Features (X) and Target (y)
# X will be all columns except 'target'
X = df_cancer.drop('target', axis=1)
y = df_cancer['target']

# Split the data into training and testing sets
# We train the model on the training data and evaluate it on the unseen testing data.
# test_size=0.2 means 20% of data for testing, 80% for training
# random_state ensures reproducibility of the split
# stratify=y ensures that the proportion of target classes is the same in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Shape of X_train: (455, 30)
# Shape of X_test: (114, 30)
# Shape of y_train: (455,)
# Shape of y_test: (114,)

# Feature Scaling: Standardise the features
# Many machine learning algorithms perform better when numerical input variables are scaled
# to a standard range. Logistic Regression benefits from this.
# StandardScaler transforms the data such that each feature (column) will have a mean of 0 and a standard deviation of 1. This is also known as Z-score normalisation.
# The formula for this transformation for a single data point x in a feature column is:
        # x_scaled = (x - μ) / σ
    # where:
     # μ: mean of the feature
     # σ: sd
 
scaler = StandardScaler() 

# Fit the scaler ONLY on the training data to learn scaling parameters
# It calculates the mean (μ) and standard deviation (σ) for each feature (column), using only the data present in X_train. It "learns" these scaling parameters.
    # These calculated means and standard deviations are stored internally within the scaler object.
scaler.fit(X_train)

# Apply the scaling transformation (using the means and standard deviations just learned from X_train) to both training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier viewing/manipulation if needed
    # scaler.transform(X_train) and scaler.transform(X_test) return NumPy arrays
# (Optional, but good for understanding what transformed data looks like)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# These are the z-scores for each original data point
# For each original value x:
        # z-score = (x - mean_of_feature) / standard_deviation_of_feature
    # If an original value was equal to the feature's mean, its scaled value (z-score) will be 0.
    # If an original value was one standard deviation above the mean, its scaled value will be 1.
    # If an original value was one standard deviation below the mean, its scaled value will be -1.
    # Values further from the mean (outliers) will have larger absolute z-scores.
# This transformation preserves the original distribution's shape but shifts and scales it.
print("\n--- Scaled X_train (First 5 rows) ---")
print(X_train_scaled_df.head())

# --- Scaled X_train (First 5 rows) ---
#      mean radius  mean texture  mean perimeter  mean area  mean smoothness  ...  worst compactness  worst concavity  worst concave points  worst symmetry  worst fractal dimension
# 546    -1.072001     -0.658425       -1.088080  -0.939274        -0.135940  ...          -1.038836        -1.078995             -1.350527       -0.352658                -0.541380
# 432     1.748743      0.066502        1.751157   1.745559         1.274468  ...           0.249655         1.178594              1.549916        0.191078                -0.173739
# 174    -0.974734     -0.931124       -0.997709  -0.867589        -0.613515  ...          -1.167260        -1.282241             -1.707442       -0.307734                -1.213033
# 221    -0.145103     -1.215186       -0.123013  -0.253192         0.664482  ...           0.096874        -0.087521             -0.344838        0.242198                -0.118266
# 289    -0.771617     -0.081211       -0.803700  -0.732927        -0.672282  ...          -0.984612        -0.933190             -0.777604        0.555118                -0.761639

# [5 rows x 30 columns]

print("\n--- Mean of Scaled X_train (should be close to 0) ---")
print(X_train_scaled_df.mean().head())

# --- Mean of Scaled X_train (should be close to 0) ---
# mean radius       -2.928061e-16
# mean texture       6.246530e-16
# mean perimeter    -1.629954e-16
# mean area         -1.717796e-16
# mean smoothness    6.246530e-17
# dtype: float64

print("\n--- Std Dev of Scaled X_train (should be close to 1) ---")
print(X_train_scaled_df.std().head())

# --- Std Dev of Scaled X_train (should be close to 1) ---
# mean radius        1.001101
# mean texture       1.001101
# mean perimeter     1.001101
# mean area          1.001101
# mean smoothness    1.001101
# dtype: float64

print("-"*100)

# ----- 3. Model Building and Training -----

# Create an instance of the Logistic Regression model
# solver='liblinear' is a good choice for smaller datasets and handles L1/L2 regularisation well
    #  "solver" refers to the optimisation algorithm used by the model to find the best parameters (coefficients and intercept) that minimise the error (or maximise the likelihood, in the case of Logistic Regression)
    # 'Regularisation' is a technique used to prevent overfitting (when a model learns the training data too well, including its noise, and performs poorly on unseen data)
    # L1 (Lasso) and L2 (Ridge) regularisation add a penalty term to the optimisation function, discouraging overly large coefficients. This effectively simplifies the model and makes it generalise better.
    # scikit-learn offers other solvers like 'lbfgs' (default, good for larger datasets, no L1), 'newton-cg', 'sag', 'saga' (good for very large datasets, supports L1/L2)
# random_state for reproducibility
model_lr = LogisticRegression(solver='liblinear', random_state=42)  
# The random_state numbers do NOT have to be the same between train_test_split and LogisticRegression
    #  Common practice in tutorials and reproducible research to use the same random_state (like 42 here) across the entire script

# Train the model using the scaled training data
# The .fit() method finds the best coefficients (weights) for the linear combination that the sigmoid function will then use to predict probabilities.
model_lr.fit(X_train_scaled, y_train)

print("\n--- Logistic Regression Model Trained ---")
print(f"Number of features: {model_lr.n_features_in_}") # Check to ensure the model received the number of features you expected (as model_lr.n_features_in_ is an attribute of scikit-learn estimators that, after fitting, tells you the number of features (columns) that the model was trained on)
print(f"Model coefficients (first 5): {model_lr.coef_[0, :5]}") # Coefficients for each feature
print(f"Model intercept: {model_lr.intercept_[0]:.4f}") # The intercept

# Number of features: 30    # (the number of columns in X_train_scaled (i.e. we had 31 columns in the dataset but dropped 'target' to create the X variable, which was then scaled))
# Model coefficients (first 5): [-0.49311166 -0.55620444 -0.46098799 -0.54817066 -0.19610022]
# Model intercept: 0.2473


print("-"*100)


### ----- 4. Prediction -----

# Predict probabilities for the test set
# .predict_proba() returns an array where each row is an instance,
# and columns are probabilities for each class (e.g., [prob_class_0, prob_class_1])
y_prob = model_lr.predict_proba(X_test_scaled)
print("\n--- Predicted Probabilities (First 5 for Class 1) ---")
print(y_prob[:5, 1]) # Print probabilities for the positive class (class 1)
    # y_prob is a 2D NumPy array.
        # The rows of y_prob correspond to the instances (samples/rows) in the X_test_scaled input.
        # The columns of y_prob correspond to the probabilities for each class.
            # Column 0 (y_prob[:, 0]) contains the probability of an instance belonging to class 0 (malignant).
            # Column 1 (y_prob[:, 1]) contains the probability of an instance belonging to class 1 (benign).
    # So, y_prob[:5, 1] means:
        # [:5]: "The first 5 rows (instances) of the y_prob array."
        # [, 1]: "And for those 5 rows, the value from the second column (index 1), which is the probability of being in class 1 (benign)."
    # Therefore, it's printing the probabilities of being 'benign' for the first 5 samples in the test set.

# --- Predicted Probabilities (First 5 for Class 1) ---
# [5.30434015e-08 9.99988047e-01 6.00279146e-03 5.29220660e-01
#  5.76439984e-10]

# Predict class labels for the test set (0 or 1)
# This uses the default threshold of 0.5 on the probabilities.
y_pred = model_lr.predict(X_test_scaled)
print("\n--- Predicted Class Labels (First 10) ---")
print(y_pred[:10])

# --- Predicted Class Labels (First 10) ---
# [0 1 0 1 0 1 1 0 0 0]

print("\n--- Actual Class Labels (First 10) ---")
print(y_test.values[:10])

# --- Actual Class Labels (First 10) ---
# [0 1 0 1 0 1 1 0 0 0]


print("-"*100)


### ----- 5. Model Evaluation -----

# Accuracy Score: The proportion of correctly classified instances
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Accuracy ---")
print(f"Accuracy: {accuracy:.4f}")

# --- Model Accuracy ---
# Accuracy: 0.9825

# Confusion Matrix: A table showing correct and incorrect classifications
# Rows represent actual classes, columns represent predicted classes.
# [[True Negative (TN), False Positive (FP)],
#  [False Negative (FN), True Positive (TP)]]
cm = confusion_matrix(y_test, y_pred)
print(f"\n--- Confusion Matrix ---")
print(cm)
print(f"True Negatives (TN): {cm[0,0]}") # Correctly predicted negative (0)
print(f"False Positives (FP): {cm[0,1]}") # Actual negative (0) but predicted positive (1) - Type I error
print(f"False Negatives (FN): {cm[1,0]}") # Actual positive (1) but predicted negative (0) - Type II error
print(f"True Positives (TP): {cm[1,1]}")  # Correctly predicted positive (1)


# --- Confusion Matrix ---
# [[41  1]
#  [ 1 71]]
# True Negatives (TN): 41       # Correctly predicted negative (0)
# False Positives (FP): 1       # Actual negative (0) but predicted positive (1) - Type I error
# False Negatives (FN): 1       # Actual positive (1) but predicted negative (0) - Type II error
# True Positives (TP): 71       # Correctly predicted positive (1)

# Classification Report: Provides Precision, Recall, F1-score for each class
# Precision: Of all predicted positives, how many were truly positive? (TP / (TP + FP))
# Recall (Sensitivity): Of all actual positives, how many were correctly predicted as positive? (TP / (TP + FN))
# F1-Score: Harmonic mean of Precision and Recall (balances both)
            # H = n / ((1/x1) + (1/x2) + (1/x3) + ... + (1/xn))
    # The harmonic mean is used because it penalises extreme values more heavily
print(f"\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


# --- Classification Report ---
#               precision    recall  f1-score   support

#            0       0.98      0.98      0.98        42
#            1       0.99      0.99      0.99        72

#     accuracy                           0.98       114
#    macro avg       0.98      0.98      0.98       114
# weighted avg       0.98      0.98      0.98       114

# Interpretation of Coefficients (Log-Odds):
# For Logistic Regression, coefficients are in terms of log-odds.
# An increase of 1 unit in a feature increases the log-odds of the positive class by the coefficient value.
# To convert to odds ratio: exp(coefficient)
# For example, if a coefficient is 0.5, exp(0.5) approx 1.65.
# This means for a one-unit increase in that feature, the odds of being in the positive class increase by ~65%.
print("\n--- Model Coefficients (Interpreting Log-Odds) ---")
# To make it more readable, we can pair coefficients with feature names
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model_lr.coef_[0]})    
    # coef_ is a 2D array where the first (and only) row [0] contains the coefficients for each feature.
    # coef_df will be a DataFrame with two columns: 'Feature' (containing names like 'mean radius') and 'Coefficient' (containing the learned numerical slope for that feature).
coef_df['Odds_Ratio'] = np.exp(coef_df['Coefficient'])
    # coef_df now has a third column named 'Odds_Ratio', where each value is e raised to the power of the corresponding coefficient.
print(coef_df.sort_values(by='Odds_Ratio', ascending=False).head()) # Features with strongest POSTIVE impact on odds of being in the positive class (benign).
print(coef_df.sort_values(by='Odds_Ratio', ascending=True).head()) # Features with strongest NEGATIVE impact on odds of being in the positive class (benign).

# Odds are the ratio of the probability of an event happening to the probability of it not happening.
        # Odds = P(event) / P(not event)
    # If P(event) = 0.75, then P(not event) = 0.25. Odds = 0.75 / 0.25 = 3. This means 3 times more likely to win than to lose.
    # If P(event) = 0.2, then P(not event) = 0.8. Odds = 0.2 / 0.8 = 0.25. (1 to 4 odds).

# Log odds.
        # Log-odds (or logit) = log(Odds) = log(P(event) / P(not event))

# Since the coefficient is a change in log-odds, if we want to understand the effect on the odds themselves, we need to "undo" the logarithm using the exponential function (exp).
        # Odds Ratio = exp(coefficient)

# Interpretation of Odds Ratio:
    # Odds Ratio = 1: The feature has no effect on the odds of the positive class.
    # Odds Ratio > 1: A one-unit increase in the feature multiplies the odds of the positive class by this amount.
        # Example: If coefficient = 0.5, then odds_ratio = exp(0.5) ≈ 1.65. This means for every one-unit increase in that feature (e.g., if a patient's 'mean radius' increases by 1 unit, holding all other features constant), the odds of them having benign cancer (our positive class) are multiplied by 1.65 (or increase by 65%).
    # Odds Ratio < 1: A one-unit increase in the feature multiplies the odds of the positive class by this amount (meaning the odds decrease).
        # Example: If coefficient = -0.5, then odds_ratio = exp(-0.5) ≈ 0.61. This means for every one-unit increase in that feature, the odds of having benign cancer are multiplied by 0.61 (or decrease by 39%). This indicates a negative association.
# i.e.:
    # Odds Ratio > 1: The feature makes it more likely.
    # Odds Ratio < 1: The feature makes it less likely.
    # The further it is from 1, the stronger the impact.

# --- Model Coefficients (Interpreting Log-Odds) ---
#                     Feature  Coefficient  Odds_Ratio
# 5          mean compactness     0.660985    1.936699
# 15        compactness error     0.650599    1.916689
# 19  fractal dimension error     0.423388    1.527127
# 18           symmetry error     0.349244    1.417995
# 11            texture error     0.251621    1.286108
#           Feature  Coefficient  Odds_Ratio
# 21  worst texture    -1.242272    0.288727
# 10   radius error    -1.087929    0.336914
# 23     worst area    -0.979282    0.375581
# 13     area error    -0.958096    0.383623
# 20   worst radius    -0.946000    0.388291
