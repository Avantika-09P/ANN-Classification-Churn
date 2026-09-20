# 🏦 Bank Customer Churn Prediction & Risk Analysis using ANN

An **Artificial Neural Network (ANN)** based machine learning application for predicting bank customer churn and analyzing customer-level churn risk.

The project combines **binary classification, churn probability prediction, risk analysis, model evaluation, and Explainable AI (SHAP)** into an interactive **Streamlit** web application.

## 🚀 Live Demo

**Streamlit Application:**
https://ann-classification-churn-57wd6awignqlwknbu6jkup.streamlit.app/

**GitHub Repository:**
https://github.com/Avantika-09P/ANN-Classification-Churn

---

## 📌 Project Overview

Customer churn occurs when a customer stops using a bank's services or closes their account. Identifying customers who may be at risk of churn can help support data-driven customer retention strategies.

This project uses an **Artificial Neural Network (ANN)** to perform binary classification and predict the probability that a customer will churn.

The application provides:

* ✅ **Retained Customer** prediction
* ⚠️ **Potential Churn** prediction
* 📊 Churn probability
* 🔍 Customer risk analysis
* 💡 SHAP-based model explainability
* 📈 Interactive Streamlit interface

---

## 🎯 Objectives

* Predict whether a bank customer is likely to churn.
* Estimate the probability of customer churn.
* Analyze factors associated with customer churn.
* Build an ANN-based binary classification model.
* Evaluate classification performance using standard metrics.
* Provide an interactive prediction interface.
* Use **SHAP** to improve model interpretability.
* Support customer-level churn risk analysis.

---

## 🧠 Machine Learning Approach

The project follows the following machine learning workflow:

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Feature Encoding & Scaling
       ↓
Train-Test Split
       ↓
Artificial Neural Network
       ↓
Churn Probability
       ↓
Churn Classification
       ↓
Risk Analysis
       ↓
SHAP Explainability
       ↓
Streamlit Deployment
```

### Artificial Neural Network

The ANN learns the relationship between customer attributes and their likelihood of churning.

Since churn prediction is a **binary classification** problem, the final layer produces a probability between 0 and 1 using the **sigmoid activation function**:

$$
P(\text{Churn}) = \sigma(z)
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

A probability closer to **1** represents a higher predicted likelihood of churn, while a probability closer to **0** represents a lower predicted likelihood.

---

## 📊 Dataset Features

The model uses customer demographic, financial, and account-related information, including:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Account Balance
* Number of Products
* Credit Card Status
* Active Membership Status
* Estimated Salary

### Target Variable

```text
Exited
```

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Customer did not churn |
| `1`   | Customer churned       |

---

## 🔍 Churn Risk Analysis

The predicted churn probability can be used to understand the relative churn risk of an individual customer.

Conceptually:

```text
Lower Churn Probability
          ↓
     Lower Risk


Higher Churn Probability
          ↓
     Higher Risk
```

This allows the application to provide more information than a simple `Yes/No` churn prediction.

---

## 💡 Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to interpret the model's predictions.

SHAP helps answer:

> **Why did the model make this prediction?**

For an individual customer, SHAP values indicate how different input features contribute to increasing or decreasing the predicted churn probability.

For example:

```text
Customer Feature
       ↓
SHAP Contribution
       ↓
Increase / Decrease in
Predicted Churn Probability
```

This provides greater transparency into the ANN's predictions and helps reduce the black-box nature of neural network models.

---

## 📈 Model Evaluation

The classification model can be evaluated using standard classification metrics.

### Accuracy

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy measures the proportion of correctly classified customers.

### Precision

$$
Precision = \frac{TP}{TP+FP}
$$

Precision measures how many customers predicted as churners actually churned.

### Recall

$$
Recall = \frac{TP}{TP+FN}
$$

Recall measures how many of the actual churners were correctly identified.

### F1 Score

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

F1-score provides a balance between precision and recall.

For churn prediction, **recall is an important metric** because failing to identify an actual churner may result in a missed opportunity for customer retention.

---

## 🖥️ Streamlit Application

The deployed application provides an interactive interface where users can enter customer information and obtain a model prediction.

### Key Features

* 👤 Customer information input
* 🔮 Churn prediction
* 📊 Churn probability
* ⚠️ Risk interpretation
* 💡 SHAP-based explanation
* 📈 Model insights
* 🌐 Web-based deployment using Streamlit

---

## 🛠️ Tech Stack

### Programming

* **Python**

### Machine Learning

* **TensorFlow / Keras**
* **Scikit-learn**

### Data Processing

* **Pandas**
* **NumPy**

### Explainable AI

* **SHAP**

### Visualization

* **Matplotlib**
* **Plotly**

### Web Application & Deployment

* **Streamlit**

### Development Tools

* **Jupyter Notebook**
* **VS Code**
* **Git**
* **GitHub**

---

## 📂 Repository

**GitHub:**
https://github.com/Avantika-09P/ANN-Classification-Churn

The repository contains the source code, machine learning components, dependencies, and supporting project files required to run the application.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Avantika-09P/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Main Python Libraries

```text
TensorFlow
Keras
Scikit-learn
Pandas
NumPy
Streamlit
SHAP
Matplotlib
Plotly
```

---

## 🔬 Project Workflow

### 1. Data Collection

A bank customer dataset containing demographic, financial, and account-related information is used for churn prediction.

### 2. Data Preprocessing

The dataset is prepared for machine learning by handling categorical and numerical features and applying the required transformations.

### 3. Model Training

An **Artificial Neural Network** is trained to learn patterns associated with customer churn.

### 4. Churn Prediction

The trained model produces a probability representing the predicted likelihood of customer churn.

### 5. Risk Analysis

The predicted probability is used to interpret the customer's relative churn risk.

### 6. Explainability

SHAP is used to analyze the contribution of individual features to the model's prediction.

### 7. Deployment

The trained model is integrated into a **Streamlit** application and deployed as an interactive web application.

---

## 🏦 Business Use Case

A bank can use a churn prediction system as a **decision-support tool** for customer retention analysis.

Potential applications include:

* Customer retention analysis
* Identification of potentially high-risk customers
* Customer segmentation
* Data-driven retention planning
* Understanding factors associated with customer churn

The predictions should be considered alongside business rules and other customer information rather than being treated as an automatic decision.

---

## 📚 Key Learning Outcomes

This project demonstrates practical implementation of:

* Binary classification
* Artificial Neural Networks
* Data preprocessing
* Feature encoding and scaling
* Model training
* Classification metrics
* Churn probability prediction
* Risk analysis
* Explainable AI
* SHAP analysis
* Streamlit application development
* Machine learning deployment

---

## 🔮 Future Enhancements

Possible future improvements include:

* Compare ANN with Logistic Regression, SVM, Random Forest, and XGBoost.
* Add more detailed SHAP visualizations.
* Introduce Customer Lifetime Value (CLV) analysis.
* Develop more granular customer risk categories.
* Add retention recommendation modules.
* Perform systematic hyperparameter optimization.
* Add model monitoring and drift detection.
* Build a complete production-oriented ML pipeline.

---

## 👩‍💻 Author

**Avantika Padhi**

B.Tech — Computer Science & Technology

---

## 🌐 Try the Application

**Live Streamlit App:**
https://ann-classification-churn-57wd6awignqlwknbu6jkup.streamlit.app/

**GitHub Repository:**
https://github.com/Avantika-09P/ANN-Classification-Churn

---

⭐ If you find this project useful, consider giving the repository a star!
