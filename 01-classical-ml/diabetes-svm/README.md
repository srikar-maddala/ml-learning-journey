# 🩺 Diabetes Prediction using Support Vector Machine (SVM)

## 📌 Project Overview

This project predicts whether a person is diabetic or non-diabetic using Machine Learning.

A Support Vector Machine (SVM) classifier is trained on the **Pima Indians Diabetes Dataset** to classify patients based on medical measurements such as glucose level, blood pressure, BMI, insulin level, and age.

The project demonstrates the complete Machine Learning workflow including data analysis, preprocessing, model training, and evaluation.

---

## 📂 Dataset

**Dataset:** Pima Indians Diabetes Dataset

The dataset contains medical information of female patients and the target variable indicates whether the patient has diabetes.

### Features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Target Variable:

- **0** → Non-Diabetic
- **1** → Diabetic

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 🔄 Machine Learning Workflow

### 1. Data Collection and Analysis

- Loaded the diabetes dataset using Pandas
- Checked dataset shape and statistical information
- Analyzed class distribution
- Studied feature relationships

---

### 2. Data Preprocessing

- Separated features and target variable
- Standardized the input features using `StandardScaler`

Why Standardization?

SVM models are sensitive to feature scales. Standardization ensures all features contribute equally to the distance calculations.

---

### 3. Train-Test Split

The dataset was divided into:

- Training data → 80%
- Testing data → 20%

Used:

```python
train_test_split()
```

with stratification to maintain class distribution.

---

## 🤖 Machine Learning Model

### Support Vector Machine (SVM)

Algorithm used:

```
Support Vector Classifier
```

Kernel:

```
Linear Kernel
```

SVM works by finding the optimal hyperplane that separates different classes.

---

## 📊 Model Evaluation

The model performance was evaluated using:

- Accuracy Score

### Training Accuracy:

```
78.66%
```

### Testing Accuracy:

```
77.27%
```

---

## 📁 Project Structure

```
diabetes_prediction_svm/
│
├── Diabetes_prediction.ipynb
├── diabetes.csv
├── README.md
```

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Compare SVM performance with:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Add confusion matrix and classification report
- Deploy the model using Streamlit
- Create an API using FastAPI

---

## 📚 Key Concepts Learned

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature standardization
- Train-test splitting
- Support Vector Machine algorithm
- Model evaluation

---

## 👨‍💻 Author

**Srikar Maddala**

Master's Student - Artificial Intelligence & Robotics
