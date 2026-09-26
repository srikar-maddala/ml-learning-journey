# 📚 ML Learning Journey

Hands-on practice projects from my path into machine learning and AI, from classical ML to deep learning,
transformers and LLMs. Each folder is a self-contained notebook project with its own README.

> For a complete end-to-end project (multimodal RAG, local LLM, evaluation, Docker, CI/CD), see
> **[brain-mri-multimodal-rag](https://github.com/srikar-maddala/brain-mri-multimodal-rag)**.

## Projects

| # | Project | Topic | Model / Technique | Result |
|---|---|---|---|---|
| 1 | [Credit card fraud detection](01-classical-ml/credit-card-fraud) | Classical ML, class imbalance | Logistic regression + undersampling | 89.3% test accuracy |
| 2 | [Diabetes prediction](01-classical-ml/diabetes-svm) | Classical ML, feature scaling | SVM + StandardScaler (Pima dataset) | 77.3% test accuracy |
| 3 | [Customer churn prediction](01-classical-ml/customer-churn) | Classical ML, categorical features | Random forest + one-hot encoding | 99.9% test accuracy ⚠️ |
| 4 | [Fashion-MNIST classification](02-deep-learning/fashion-mnist) | Deep learning fundamentals | PyTorch MLP, custom Dataset/DataLoader | 83.3% test accuracy |
| 5 | [IMDb sentiment classification](03-nlp-and-llms/bert-imdb-sentiment) | NLP, transfer learning | Fine-tuned DistilBERT (Hugging Face Trainer) | Eval loss 0.35 |
| 6 | [LLM inference](03-nlp-and-llms/llm-inference-huggingface) | LLMs, text generation | LiquidAI LFM2.5-2.6B, sampling parameters | Qualitative experiments |

## What I learned

- **Class imbalance:** fraud cases are rare, so accuracy alone is misleading; undersampling was a first fix, and
  precision, recall and ROC-AUC are the better metrics.
- **Feature scaling matters** for distance-based models like SVMs.
- **Overfitting:** the Fashion-MNIST network reached 99.6% training but 83.3% test accuracy, a clear sign that
  regularization (dropout, weight decay) or a CNN is needed.
- **Too-good results need checking:** 99.9% on churn is suspicious and should be verified on the separate
  test file with per-class metrics.
- **Transfer learning:** fine-tuning a pretrained DistilBERT gets strong NLP results with little code.
- **How LLMs generate text:** tokenization, autoregressive decoding, temperature, top-k and top-p sampling.

## Tech stack

Python · NumPy · Pandas · scikit-learn · PyTorch · Hugging Face Transformers & Datasets · Matplotlib ·
Jupyter / Google Colab

## Structure

```
01-classical-ml/
  credit-card-fraud/      logistic regression, imbalanced data
  diabetes-svm/           support vector machine
  customer-churn/         random forest
02-deep-learning/
  fashion-mnist/          PyTorch neural network from scratch
03-nlp-and-llms/
  bert-imdb-sentiment/    DistilBERT fine-tuning
  llm-inference-huggingface/   LLM text generation
```
