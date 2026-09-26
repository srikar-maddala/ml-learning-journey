# 📉 Customer Churn Prediction using Random Forest

## 📌 Project Overview

Customer churn prediction is a machine learning problem where we predict whether a customer is likely to leave a company's service.

In this project, a **Random Forest Classifier** is used to predict customer churn based on customer demographic information, usage behavior, and subscription details.

The goal of this project is to identify customers who are likely to churn so that businesses can take preventive actions such as offering better services, discounts, or customer support.

---

## 🎯 Problem Statement

Companies spend significant resources acquiring customers. Losing existing customers affects revenue.

Using machine learning, this project predicts:

- **0 → Customer stays (No Churn)**
- **1 → Customer leaves (Churn)**

---

## 📂 Dataset

The dataset contains customer information including:

### Features

### Customer Information
- Customer ID
- Gender
- Age

### Subscription Information
- Subscription Type
- Contract Length

### Customer Behaviour
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction

### Target Variable

- **Churn**

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook
- Matplotlib

---

## 🔄 Machine Learning Workflow

### 1. Data Loading

- Loaded customer churn dataset using Pandas.
- Checked dataset structure and missing values.

---

### 2. Data Cleaning

- Removed rows containing missing values in the target column.
- Separated input features and target variable.

---

### 3. Feature Engineering

Categorical variables were converted into numerical format using:

```
OneHotEncoder
```

Encoded features:

- Gender
- Subscription Type
- Contract Length

Numerical features used:

- Age
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction

---

### 4. Train-Test Split

The dataset was divided into:

- Training data → 80%
- Testing data → 20%

Using:

```python
train_test_split()
```

---

## 🤖 Machine Learning Model

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

Advantages:

- Handles nonlinear relationships
- Reduces overfitting compared to a single decision tree
- Provides feature importance
- Works well with mixed data types

---

## 📊 Model Evaluation

The model was evaluated using:

### Accuracy Score

```python
accuracy_score()
```

Additional evaluation metrics:

- Confusion Matrix
- Precision
- Recall
- F1-score

---

## 📈 Results

Model performance:

```
Accuracy: 99.93%
```

> ⚠️ This is suspiciously high. The features may almost perfectly separate the classes in this
> dataset. A next step is to evaluate on the separate `customer_churn_dataset-testing-master.csv`
> file and report precision, recall and F1 per class.

---

## 📊 Visualizations

The project includes:

### Confusion Matrix

Shows:

- Correct predictions
- Incorrect predictions
- Churn prediction performance

### Feature Importance

Shows which customer attributes have the highest influence on churn prediction.

---

## 📁 Project Structure

```
customer_churn_prediction/

│
├── customer_churn_prediction.ipynb
├── customer_churn_dataset-training-master.csv
├── customer_churn_dataset-testing-master.csv
├── README.md
```

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Compare with:
  - Logistic Regression
  - Decision Tree
  - XGBoost
  - Neural Networks
- Handle class imbalance using SMOTE
- Deploy model using Streamlit
- Create API using FastAPI

---

## 📚 Key Concepts Learned

- Data preprocessing
- One-hot encoding
- Feature engineering
- Random Forest classification
- Model evaluation
- Business problem solving using Machine Learning

---

## 👨‍💻 Author

**Srikar Maddala**

Master's Student - Artificial Intelligence & Robotics
