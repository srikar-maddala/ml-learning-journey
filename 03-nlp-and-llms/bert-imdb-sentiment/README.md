# BERT IMDb Sentiment Classification using DistilBERT

## Project Overview

This project demonstrates fine-tuning a pretrained Transformer model for binary sentiment classification.

A pretrained **DistilBERT** model is fine-tuned on the **IMDb Movie Reviews dataset** to classify movie reviews into:

* **0 → Negative Sentiment**
* **1 → Positive Sentiment**

The project uses **PyTorch** and **Hugging Face Transformers** for model training and evaluation.

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* DistilBERT
* Scikit-learn
* Google Colab GPU

---

# Project Workflow

```
IMDb Dataset
      |
      ↓
Text Preprocessing
      |
      ↓
DistilBERT Tokenization
      |
      ↓
Input IDs + Attention Masks
      |
      ↓
DistilBERT Model
      |
      ↓
Fine-tuning using Hugging Face Trainer
      |
      ↓
Evaluation
      |
      ↓
Sentiment Prediction
```

---

# Dataset

## IMDb Movie Reviews Dataset

The IMDb dataset contains movie reviews with sentiment labels.

Dataset features:

| Feature | Description       |
| ------- | ----------------- |
| text    | Movie review text |
| label   | Sentiment label   |

Labels:

| Label | Meaning         |
| ----- | --------------- |
| 0     | Negative Review |
| 1     | Positive Review |

---

# Model Architecture

## DistilBERT

This project uses:

```
distilbert-base-uncased
```

DistilBERT is a smaller and faster version of BERT while maintaining most of BERT's performance.

Architecture:

```
Input Text
    |
    ↓
Tokenizer
    |
    ↓
Input IDs + Attention Mask
    |
    ↓
DistilBERT Encoder
    |
    ↓
Classification Layer
    |
    ↓
Positive / Negative Prediction
```

---

# Tokenization

Text cannot be directly given to a Transformer model.

The tokenizer converts text into numerical representations.

Example:

```
"This movie was amazing"
```

becomes:

```
input_ids:
[101, 2023, 3185, 2001, 6429, 102]

attention_mask:
[1,1,1,1,1,1]
```

The model uses:

* `input_ids` → token information
* `attention_mask` → identifies real tokens and padding tokens

---

# Fine-tuning Process

The pretrained DistilBERT model already understands language from large-scale training.

During fine-tuning:

1. IMDb reviews are provided as input.
2. DistilBERT generates representations.
3. Classification head predicts sentiment.
4. Loss is calculated.
5. Model weights are updated using backpropagation.

Training was performed using Hugging Face:

```
Trainer API
```

---

# Training Configuration

Example configuration:

```
Model:
DistilBERT

Task:
Binary Sentiment Classification

Optimizer:
AdamW (Hugging Face Trainer default)

Epochs:
3

Batch Size:
16

Learning Rate:
2e-5
```

---

# Evaluation

The trained model is evaluated on unseen test data.

Evaluation metrics:

* Validation Loss
* Accuracy

Example prediction:

```
Input:
"This movie was fantastic. I loved every moment."

Output:
Positive
```

---

# Inference Example

After fine-tuning, the model can classify new reviews.

Example:

```python
review = "The movie was excellent and entertaining"

Prediction:

Positive
```

Example:

```python
review = "The movie was boring and disappointing"

Prediction:

Negative
```

---

# Project Structure

```
BERT_IMDB_Project
│
├── BERT_IMDB_Sentiment_Classification.ipynb
│
├── README.md
│
└── .gitignore
```

---

# How to Run the Project

## Install dependencies

```bash
pip install torch transformers datasets scikit-learn pandas numpy
```

## Run Notebook

Open:

```
BERT_IMDB_Sentiment_Classification.ipynb
```

Run the cells sequentially.

---

# Key Concepts Covered

This project covers:

* Transformer architecture
* BERT and DistilBERT
* Tokenization
* Attention Mask
* Pretrained Models
* Transfer Learning
* Fine-tuning
* Hugging Face Trainer
* Model Evaluation
* Transformer Inference

---

# Future Improvements

Possible extensions:

* Deploy the model using FastAPI
* Create a web application using Streamlit
* Upload the fine-tuned model to Hugging Face Hub
* Experiment with BERT, RoBERTa, and ALBERT models

---

# Author

Srikar Maddala

AI Engineer | Machine Learning | Deep Learning | Transformers
