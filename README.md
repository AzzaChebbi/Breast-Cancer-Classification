# 🧠 Breast Cancer Classification using Machine Learning

A comprehensive machine learning project for predicting **Coimbra breast cancer** using clinical and biochemical biomarkers.  
This project compares multiple classification algorithms and implements **custom gradient descent optimization** for logistic regression.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
---

## 🎯 Overview

Breast cancer remains one of the leading causes of cancer-related deaths among women worldwide.  
This project develops a **binary classification model** to predict whether a patient has Coimbra breast cancer based on clinical and biochemical attributes.

It includes:
- Comprehensive **Exploratory Data Analysis (EDA)**
- Data preprocessing and outlier removal
- **Hyperparameter tuning**
- Custom **gradient descent implementation** for logistic regression
- Performance comparison across multiple models

---

## 📊 Dataset

The dataset contains **116 patient records** (reduced to 112 after outlier removal) with the following features:

| Feature | Description |
|----------|--------------|
| Age | Patient age (years) |
| BMI | Body Mass Index (kg/m²) |
| Glucose | Blood glucose levels (mg/dL) |
| Insulin | Serum insulin levels (µU/mL) |
| HOMA | Homeostatic Model Assessment |
| Leptin | Leptin levels (ng/mL) |
| Adiponectin | Adiponectin levels (µg/mL) |
| Resistin | Resistin levels (ng/mL) |
| MCP-1 | Monocyte Chemoattractant Protein 1 (pg/dL) |
| Classification | Target variable (0 = Healthy, 1 = Cancer) |

### 🧹 Data Preprocessing
- Outlier detection and removal (MCP-1 values > 1500)
- Feature standardization using `StandardScaler`
- Train-test split (75% - 25%)

---

## 🤖 Models Implemented

### 1️⃣ K-Nearest Neighbors (KNN)
- Initial accuracy: 67.86%  
- After hyperparameter tuning: **75%**  
- Best parameters: `n_neighbors=7`, `metric='manhattan'`, `weights='distance'`

### 2️⃣ Logistic Regression
- Initial accuracy: 82.14%  
- After tuning: **85.71%**  
- Custom gradient descent implementation: **96.43%**  
- Best parameters: `C=0.954`, `penalty='l1'`, `solver='liblinear'`

### 3️⃣ Naive Bayes
- Initial accuracy: 67.86%  
- After tuning: **89.29%**  
- Best parameter: `var_smoothing=0.231`

### 4️⃣ Neural Network (PyTorch)
- Architecture: Single-layer perceptron with sigmoid activation  
- Optimizer: Adam with L2 regularization  
- Accuracy: **75%**  
- Training: 2000 epochs

---

## 📈 Results

### Model Performance Comparison

| Model | Initial Accuracy | Tuned Accuracy | AUC-ROC |
|--------|------------------|----------------|----------|
| KNN | 67.86% | 75.00% | 0.740 |
| Logistic Regression | 82.14% | 85.71% | 0.929 |
| **Logistic Regression (GD)** | - | **96.43%** | - |
| Naive Bayes | 67.86% | 89.29% | 0.635 |
| Neural Network | - | 75.00% | - |

### 🏆 Best Model
**Logistic Regression with Gradient Descent** achieved the highest accuracy of **96.43%** on the test set.

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

## 💻 Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2️⃣ Prepare the Dataset

Place the file dataR2.csv in the project’s root directory.

### 3️⃣ Run the Jupyter Notebook
jupyter notebook breast_cancer_classification.ipynb

### 4️⃣ Execute Cells Sequentially

- Load and explore the dataset
- Preprocess features
- Train machine learning models
- Evaluate performance
- Compare and visualize the results

## 📁 Project Structure
breast-cancer-classification/
│
├── dataR2.csv                          # Dataset
├── breast_cancer_classification.ipynb  # Main notebook
└── README.md                           # Project documentation

## 🔍 Key Findings

Feature Importance: Glucose, insulin, and leptin levels showed strong correlations with breast cancer classification.

Outlier Impact: Removing MCP-1 outliers (values > 1500) improved model stability.

Model Selection:

Naive Bayes performed well after hyperparameter tuning (89.29%)

Custom gradient descent on logistic regression achieved the best results (96.43%)

Hyperparameter Tuning: Crucial for improving model performance across all algorithms.

## 🛠️ Technologies Used

🐍 Python 3.x – Core programming language

📊 NumPy – Numerical computations

📈 Pandas – Data manipulation and analysis

🎨 Matplotlib & Seaborn – Data visualization

🤖 Scikit-learn – Machine learning utilities

🔥 PyTorch – Neural network implementation

🧮 SciPy – Statistical functions and optimizations

📊 Visualizations :

- Distribution plots for age and glucose
- Correlation heatmaps
- Box plots for outlier detection
- Confusion matrices for model evaluation
- ROC curves for binary classification
- Loss curves for neural network training
