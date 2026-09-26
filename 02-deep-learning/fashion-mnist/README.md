# Fashion MNIST Classification using PyTorch Neural Network

## 1. Project Overview

This project implements a **Deep Learning image classification model using PyTorch**.

The goal is to teach a Neural Network how to recognize different types of clothing images from the Fashion MNIST dataset.

The complete workflow:

```
Dataset
   |
Data Exploration
   |
Data Preprocessing
   |
Custom Dataset Creation
   |
DataLoader
   |
Neural Network
   |
Forward Propagation
   |
Loss Calculation
   |
Backpropagation
   |
Weight Update
   |
Model Evaluation
```

---

# 2. Technologies Used

- Python
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

---

# 3. Understanding the Dataset

## Fashion MNIST

Fashion MNIST is an image dataset containing clothing images.

It contains:

- 70,000 images
- Each image size: 28 × 28 pixels
- 10 different classes


Example:

```
Image

28 pixels × 28 pixels

= 784 pixel values
```


Each image is converted into numerical values.

Example:

```
[
 0, 0, 12, 45,
 200, 255, ...
]
```

The model does not understand images directly.

It only understands numbers.

---

# 4. Importing Libraries

```python
import pandas as pd
```

Pandas is used for:

- Reading CSV files
- Data manipulation
- Data analysis


Example:

```python
df = pd.read_csv("fmnist_small.csv")
```


---

```python
import torch
```

PyTorch is a Deep Learning framework.

It provides:

- Tensor operations
- Neural Network building blocks
- GPU acceleration
- Automatic differentiation


---

```python
from torch.utils.data import Dataset, DataLoader
```

Dataset and DataLoader help manage training data.

Dataset:

- Stores data
- Provides individual samples


DataLoader:

- Creates batches
- Shuffles data
- Loads data efficiently


---

```python
import torch.nn as nn
```

`torch.nn` contains neural network components.

Examples:

- Linear layers
- Activation functions
- Loss functions


---

```python
import torch.optim as optim
```

Optimizer updates neural network weights.

Examples:

- SGD
- Adam
- RMSProp


---

# 5. Random Seed

```python
torch.manual_seed(42)
```

A neural network uses random initialization.

Example:

Weights are randomly created:

```
Weight = 0.324
Weight = -0.567
```

Setting a seed makes results reproducible.

---

# 6. GPU Detection

```python
device = torch.device(
'cuda' if torch.cuda.is_available()
else 'cpu'
)
```

A GPU can perform matrix calculations much faster than CPU.

The code checks:

```
Is GPU available?

Yes → use CUDA GPU

No → use CPU
```

---

# 7. Loading Dataset

```python
df = pd.read_csv(
'fmnist_small.csv'
)
```

The CSV contains:

Example:

| Label | Pixel1 | Pixel2 | ... |
|-|-|-|-|
|5|0|0|12|
|2|1|10|50|

First column:

```
Target label
```

Remaining columns:

```
Image pixels
```

---

# 8. Image Visualization

Images are stored as 784 numbers.

But the original image is:

```
28 × 28
```

So we reshape:

```python
reshape(28,28)
```

Example:

Before:

```
784 values
```

After:

```
28 rows
28 columns
```

Matplotlib displays the image.

---

# 9. Train-Test Split

Machine learning requires:

## Training Data

Used to learn patterns.


## Testing Data

Used to check performance on unseen data.


Code:

```python
train_test_split()
```

Example:

```
80% Training

20% Testing
```

---

# 10. Feature Scaling

Pixel values:

```
0 - 255
```

Example:

```
255 = white pixel
0 = black pixel
```


We normalize:

```python
X_train = X_train / 255
```

Now:

```
0 - 1
```

Benefits:

- Faster training
- Stable gradients
- Better convergence

---

# 11. PyTorch Dataset Class


```python
class CustomDataset(Dataset):
```

We create our own dataset class.

Why?

Because PyTorch needs a standard way to access data.


It requires two methods:

## __len__()

Returns number of samples.

Example:

```python
len(dataset)
```

Output:

```
8000
```


---

## __getitem__()

Returns one sample.

Example:

```python
dataset[5]
```

Returns:

```
(image, label)
```

---

# 12. Creating Dataset Object


```python
train_dataset = CustomDataset(
X_train,
y_train
)
```

Here:

X_train:

```
Input features
```

Example:

```
pixels
```


y_train:

```
Target labels
```

Example:

```
shirt, shoe, bag
```

---

# 13. DataLoader


```python
DataLoader(
dataset,
batch_size=32
)
```

Instead of sending all data:

```
8000 images
```

we send batches:

```
Batch 1 → 32 images

Batch 2 → 32 images

Batch 3 → 32 images
```


Advantages:

- Faster training
- Less memory usage
- Better optimization

---

# 14. Neural Network


Our model:

```
Input Layer
784 neurons

        |
        ↓

Linear Layer

784 → 128

        |
        ↓

ReLU

        |
        ↓

128 → 64

        |
        ↓

ReLU

        |
        ↓

64 → 10

Output Layer
```


---

# 15. nn.Module


```python
class MyNN(nn.Module):
```


All PyTorch models inherit from:

```
nn.Module
```


It provides:

- Parameter tracking
- Gradient calculation
- Saving model
- GPU support


---

# 16. super().__init__()


```python
super().__init__()
```


Calls the parent class constructor.


Structure:

```
nn.Module

     ↑

 MyNN
```


It initializes PyTorch internal systems.

Without it:

- Layers are not registered
- Parameters cannot be optimized

---

# 17. Linear Layer


```python
nn.Linear(784,128)
```


A linear layer performs:

```
Output = Input × Weight + Bias
```


It learns relationships between pixels.

---

# 18. Activation Function


```python
nn.ReLU()
```


ReLU:

```
f(x)=max(0,x)
```


Example:

Input:

```
[-2,5,-1,8]
```


Output:

```
[0,5,0,8]
```


It adds non-linearity.

Without activation functions:

The network cannot learn complex patterns.

---

# 19. Forward Function


```python
def forward(self,x):
```

Defines how data moves through the network.


Example:

```
Input

 ↓

Layer 1

 ↓

Activation

 ↓

Layer 2

 ↓

Output
```


When we call:

```python
model(images)
```

PyTorch automatically calls:

```python
forward()
```

---

# 20. Loss Function


```python
nn.CrossEntropyLoss()
```


Loss measures:

"How wrong is my prediction?"


Example:

Actual:

```
Class = 5
```


Prediction:

```
Class = 3
```


Loss increases.


Goal:

Reduce loss.

---

# 21. Optimizer


```python
optim.SGD()
```


Optimizer updates weights.


Learning process:

```
Prediction

 ↓

Calculate Error

 ↓

Calculate Gradient

 ↓

Update Weights
```


---

# 22. Training Loop


## Forward Pass


```python
outputs=model(batch_features)
```


The model makes predictions.


---

## Calculate Loss


```python
loss=criterion(
outputs,
batch_labels
)
```


Measures error.


---

## Zero Gradients


```python
optimizer.zero_grad()
```


Removes previous gradients.


---

## Backpropagation


```python
loss.backward()
```


Calculates:

```
Which weights caused the error?
```

---

## Update Weights


```python
optimizer.step()
```


Changes weights to reduce error.

---

# 23. Evaluation


During testing:


```python
model.eval()
```


Changes model to evaluation mode.


---

```python
torch.no_grad()
```


Stops gradient calculation.

Benefits:

- Faster
- Less memory usage


---

# 24. Prediction


```python
torch.max(outputs,1)
```


Finds the class with highest score.


Example:

Output:

```
[0.1,0.2,5.6,0.4]
```


Prediction:

```
Class 2
```

---

# 25. Accuracy


Formula:

```
Accuracy =
Correct Predictions / Total Predictions
```


Example:

```
900 correct

1000 images

Accuracy = 90%
```

---

# 26. Future Improvements

Possible upgrades:

- CNN instead of ANN
- Adam optimizer
- Dropout
- Batch Normalization
- Hyperparameter tuning
- Confusion Matrix
- Transfer Learning


---

# Author

Srikar Maddala

AI & Robotics Master's Student

GitHub:

https://github.com/srikar-maddala
