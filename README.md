# Logistic Regression on Breast Cancer Dataset 🧬🔍

This project demonstrates how to implement **logistic regression** using Python and scikit-learn to classify **malignant vs. benign tumours** from the Breast Cancer Wisconsin (Diagnostic) dataset.

It is structured to teach beginners not only how to code logistic regression, but also how to interpret its predictions using concepts like **probabilities, log-odds, odds ratios**, and **confusion matrices**. The script <ins>*includes detailed comments and example outputs to support self-learning*</ins>.

---

## 📌 Project Goals

- Learn how logistic regression works for binary classification
- Apply logistic regression to a real-world medical dataset
- Practice preprocessing (scaling, splitting, encoding)
- Interpret model coefficients as odds ratios
- Evaluate model using accuracy, precision, recall, and F1-score

---

## 📊 Dataset

**Source**: Built-in dataset from `sklearn.datasets.load_breast_cancer()`  
**Instances**: 569  
**Features**: 30 numeric features describing characteristics of cell nuclei  
**Target**:  
- `0` = malignant  
- `1` = benign  

---

## 📚 Theoretical Background

### What is Logistic Regression?

Logistic Regression:
- Used for binary classification problems. It predicts the probability that an instance belongs to a certain class.

The Sigmoid Function
-  Logistic regression takes the linear combination of the features (just like in Linear Regression) and passes it through a special function called the Sigmoid (or Logistic) function

        σ(z) = 1 / (1 + e^-z)
                

    - where z is the linear combination of the features: z = beta_0 + beta_1x_1 + beta_2x_2 + ... + beta_nx_n
    - Output between 0 and 1: 
        - This output can be directly interpreted as a **probability**.
            - If sigma(z) is close to 1, it means a high probability of belonging to the positive class (benign).
            - If sigma(z) is close to 0, it means a high probability of belonging to the negative class (malignant).

### Interpreting Coefficients (Log-Odds and Odds Ratios)

- Coefficients in logistic regression represent **log-odds**.
- To interpret them in terms of impact on the odds of being benign:

      Odds Ratio = exp(coefficient)
    - Odds Ratio > 1 → the feature **increases** the chance of being benign  
    - Odds Ratio < 1 → the feature **decreases** the chance of being benign

---

## 🧪 Key Steps in the Script

### 1. **Data Loading**
- Load the dataset using `load_breast_cancer(as_frame=True)`
- Add human-readable labels for targets (`malignant`, `benign`)

### 2. **Exploration**
- View head of the DataFrame, data types, and value counts
- Get statistical summaries using `.describe()`

### 3. **Preprocessing**
- Split data into training and test sets (80/20), with stratification
- Scale features using `StandardScaler` to normalise data (Z-score)

### 4. **Model Training**
- Instantiate logistic regression with `liblinear` solver
- Fit the model on the training data

### 5. **Prediction**
- Predict both **probabilities** and **classes**
- Use a default threshold of 0.5 for classification

### 6. **Evaluation**
- Accuracy
- Confusion matrix
- Classification report (Precision, Recall, F1-score)
- Top features impacting prediction (by Odds Ratio)

---

## 📈 Example Outputs

### Classification Report

```
          precision    recall  f1-score   support

       0       0.98      0.98      0.98        42
       1       0.99      0.99      0.99        72

accuracy                           0.98       114
```

### Top Features Increasing Odds of Benign Diagnosis
```
| Feature               | Coefficient | Odds Ratio |
|-----------------------|-------------|------------|
| mean compactness      | 0.661       | 1.937      |
| compactness error     | 0.651       | 1.917      |
| fractal dimension err | 0.423       | 1.527      |
```

### Top Features Decreasing Odds of Benign Diagnosis
```
| Feature        | Coefficient | Odds Ratio |
|----------------|-------------|------------|
| worst texture  | -1.242      | 0.289      |
| radius error   | -1.088      | 0.337      |
| worst area     | -0.979      | 0.376      |
```

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

requirements.txt
```txt
matplotlib==3.10.3
numpy==2.3.1
pandas==2.3.0
scikit-learn==1.7.0
seaborn==0.13.2
```

---

## 🚀 How to Run
1.	Clone the repo
```
git clone https://github.com/adabyt/logistic_regression_tutorial.git
cd logistic_regression_tutorial
```
2.	Install requirements
```
pip install -r requirements.txt
```
3.	Run the script:
```
python logistic_regression.py
```
