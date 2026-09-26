
# 💳 Credit Card Fraud Detection using Logistic Regression

## 📌 Project Overview

This project uses a **Logistic Regression** model to detect fraudulent credit card transactions. Since fraudulent transactions are extremely rare compared to legitimate ones, the dataset is highly imbalanced. To address this issue, **undersampling** is performed by selecting an equal number of legitimate transactions and fraudulent transactions before training the model.

---

## 📂 Dataset

- **Dataset:** Credit Card Fraud Detection Dataset
- **Target Column:** `Class`
  - `0` → Legitimate Transaction
  - `1` → Fraudulent Transaction

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

---

## 📚 Machine Learning Algorithm

- Logistic Regression

---

## 🔄 Project Workflow

1. Load the dataset.
2. Explore the dataset and check for missing values.
3. Analyze the distribution of legitimate and fraudulent transactions.
4. Separate legitimate and fraudulent transactions.
5. Perform **undersampling** to balance the dataset.
6. Split the data into training and testing sets.
7. Train a Logistic Regression model.
8. Evaluate the model using accuracy score.

---

## 📊 Data Preprocessing

- Checked for missing values.
- Examined class imbalance.
- Applied random undersampling to balance the dataset.
- Used `train_test_split()` with stratification to maintain class distribution.

---

## ⚙️ Model Training

The Logistic Regression model was trained using the training dataset.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

## 📈 Model Evaluation

The model performance was evaluated using:

- Training Accuracy
- Testing Accuracy

Example:

```
Accuracy on training data: 94.54%
Accuracy on testing data: 89.34%
```

*(Replace these values with your actual results.)*

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── creditcard.csv
├── Credit_Card_Fraud_Detection.ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🚀 Future Improvements

- Apply StandardScaler to numerical features.
- Compare Logistic Regression with:
  - Decision Tree
  - Random Forest
  - XGBoost
- Perform Hyperparameter Tuning.
- Use Cross Validation.
- Evaluate using Precision, Recall, F1-score, and ROC-AUC instead of relying only on accuracy.
- Handle the imbalanced dataset using **SMOTE** instead of undersampling.

---

## 📖 Key Concepts Learned

- Data preprocessing
- Handling imbalanced datasets
- Undersampling
- Train-Test Split
- Logistic Regression
- Model Evaluation
- Binary Classification

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook or Python script.

---

## 📌 Author

**Srikar Maddala**

Master's in Artificial Intelligence and Robotics

Interested in Machine Learning, Deep Learning, LLMs, and AI Software Engineering.
Logistic_regression_credit_card

