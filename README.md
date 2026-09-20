# 🏦 Bank Customer Churn Prediction & Risk Analysis using ANN

An **Artificial Neural Network (ANN)** based machine learning application that predicts whether a bank customer is likely to **churn** and provides insights into the factors associated with customer attrition.

The project combines **customer churn classification, risk analysis, model evaluation, and explainable AI (SHAP)** into an interactive Streamlit dashboard.

## 🚀 Live Demo

🔗 **Streamlit App:**
https://ann-classification-churn-57wd6awignqlwknbu6jkup.streamlit.app/

---

## 📌 Project Overview

Customer churn occurs when a customer stops using a bank's services or closes their account. Predicting churn in advance can help banks identify customers who may be at risk and take appropriate retention measures.

This project uses an **Artificial Neural Network (ANN)** to classify customers into:

* ✅ **Retained Customer**
* ⚠️ **Potential Churn Customer**

The application also provides customer-level predictions and model explanations to understand which features contribute to the prediction.

---

## 🎯 Objectives

* Predict whether a customer is likely to churn.
* Analyze important factors affecting customer churn.
* Use an ANN for binary classification.
* Evaluate the model using multiple performance metrics.
* Provide an interactive interface for customer-level predictions.
* Use **SHAP** for model interpretability.
* Categorize customers based on their predicted churn risk.

---

## 🧠 Machine Learning Approach

The project follows a standard machine learning pipeline:

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
Risk Classification
       ↓
SHAP Explainability
       ↓
Interactive Streamlit Dashboard
```

### Artificial Neural Network

The ANN learns the relationship between customer attributes and the probability of churn.

The model performs **binary classification**, where the output represents the probability that a customer will churn.

A sigmoid activation function is used for the final output:

$$
P(\text{Churn}) = \sigma(z)
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

A probability closer to **1** indicates a higher likelihood of churn, while a probability closer to **0** indicates a lower likelihood.

---

## 📊 Dataset Features

The model uses customer-related attributes such as:

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

The target variable is:

```text
Exited
```

where:

* `0` → Customer did not churn
* `1` → Customer churned

---

## 🔍 Risk Analysis

Instead of only returning a binary churn prediction, the application can use the predicted churn probability to understand customer risk.

Conceptually:

```text
Low Churn Probability
        ↓
Lower Risk

High Churn Probability
        ↓
Higher Risk
```

This makes the system more useful for customer retention analysis because banks can focus their attention on customers with comparatively higher predicted churn risk.

---

## 💡 Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to improve model interpretability.

SHAP helps answer:

> **"Why did the model make this prediction?"**

For an individual customer, SHAP values indicate how different features contribute toward increasing or decreasing the predicted churn probability.

For example:

```text
Feature                 Effect
--------------------------------
High Age                ↑ Churn Risk
Inactive Member         ↑ Churn Risk
Multiple Products       ↓/↑ Risk
High Balance            ↓/↑ Risk
```

The actual contribution depends on the customer's input values and the trained model.

This makes the ANN model easier to interpret instead of treating it as a complete black box.

---

## 📈 Model Evaluation

The classification model can be evaluated using:

### Accuracy

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

$$
Precision = \frac{TP}{TP+FP}
$$

### Recall

$$
Recall = \frac{TP}{TP+FN}
$$

### F1 Score

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

For churn prediction, **recall is particularly important** because missing a genuinely at-risk customer can prevent the bank from taking an opportunity to retain them.

---

## 🖥️ Application Features

The Streamlit application provides an interactive interface for:

* 👤 Entering customer information
* 🔮 Predicting churn probability
* 📊 Viewing the churn prediction
* ⚠️ Understanding customer risk
* 🔍 Interpreting model predictions
* 📈 Exploring model-related insights

---

## 🛠️ Tech Stack

### Programming Language

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

### Web Application

* **Streamlit**

### Development

* **Jupyter Notebook**
* **VS Code**
* **Git & GitHub**

---

## 📂 Project Structure

```text
Bank-Customer-Churn/
│
├── app.py
├── model/
│   └── churn_model.h5
│
├── data/
│   └── churn_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact file structure may vary depending on the final GitHub repository organization.

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Bank-Customer-Churn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Main Python Libraries

```text
tensorflow
keras
scikit-learn
pandas
numpy
streamlit
shap
matplotlib
plotly
```

---

## 🔬 Project Workflow

### Step 1 — Data Collection

A bank customer dataset containing demographic, financial, and account-related information is used.

### Step 2 — Data Preprocessing

The data is cleaned and prepared for machine learning.

Categorical variables are encoded and numerical features are scaled where required.

### Step 3 — Model Training

An Artificial Neural Network is trained using the processed customer data.

### Step 4 — Classification

The trained model generates a churn probability for a customer.

### Step 5 — Risk Analysis

The probability is interpreted to identify customers who may have a higher risk of churn.

### Step 6 — Explainability

SHAP is used to investigate which input features contributed to the model's prediction.

### Step 7 — Deployment

The trained model is integrated into a Streamlit application and deployed online.

---

## 🏦 Business Use Case

A bank can use a churn prediction system to identify customers who may be at risk of leaving.

The system can support:

* Customer retention analysis
* Risk identification
* Personalized retention strategies
* Customer segmentation
* Data-driven decision making

The model is intended as a **decision-support system**, not as an automatic replacement for business judgment.

---

## 📌 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Binary classification
* Artificial Neural Networks
* Feature preprocessing
* Model training and evaluation
* Classification metrics
* Churn probability prediction
* Explainable AI
* SHAP analysis
* Streamlit application development
* Machine learning model deployment

---

## 🔮 Future Enhancements

Possible improvements include:

* Compare ANN with Logistic Regression, SVM, Random Forest, and XGBoost.
* Add interactive SHAP visualizations.
* Introduce customer lifetime value (CLV) analysis.
* Develop more detailed customer risk tiers.
* Add automated retention recommendations.
* Perform hyperparameter optimization.
* Add model monitoring and drift detection.
* Deploy the model using a production ML pipeline.

---

## 👩‍💻 Author

**Avantika Padhi**

B.Tech — Computer Science & Technology

---

## 🌐 Live Application

Try the deployed application:

**https://ann-classification-churn-57wd6awignqlwknbu6jkup.streamlit.app/**

---

⭐ If you find this project useful, consider giving the repository a star!
