# Customer Churn Prediction System

End-to-End Machine Learning system for predicting customer churn using classification models, ensemble learning, and Flask deployment.

---

## Project Overview

Customer churn is one of the most important business challenges for subscription-based companies. This project predicts whether a customer is likely to leave the service using machine learning models trained on customer demographic, service usage, and billing information.

The project demonstrates the complete machine learning lifecycle including data preprocessing, feature engineering, model training, evaluation, ensemble learning, and deployment through a Flask web application.

---

## Problem Statement

Customer retention is significantly less expensive than acquiring new customers. The objective of this project is to identify high-risk customers and enable proactive retention strategies through predictive analytics.

---

## Dataset

### Source

Telco Customer Churn Dataset

### Dataset Information

* 7,000+ customer records
* 40+ features
* Customer demographics
* Account information
* Services subscribed
* Billing details
* Contract information

### Target Variable

* Churn (Yes / No)

---

## Project Workflow

### 1. Data Collection & Cleaning

* Prepared dataset for modeling
* Identified and removed constant/redundant features
* Dropped non-informative columns (e.g., ID and constant-value columns)

### 2. Exploratory Data Analysis (EDA)

* Examined feature distributions
* Identified constant and redundant features
* Performed missing value analysis
* Analyzed target class distribution
* Generated statistical summaries for feature understanding

### 3. Feature Engineering

* Encoding categorical variables
* Feature scaling
* Data transformation
* Pipeline-based preprocessing

### 4. Model Development

The following models were trained and evaluated:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Models
* Voting Classifier (Ensemble)

### 5. Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

### 6. Deployment

* Flask Web Application
* Real-time prediction interface
* Probability-based churn prediction

---

## Model Performance

| Metric              | Score                       |
| ------------------- | --------------------------- |
| ROC-AUC             | ~0.85                       |
| F1 Score            | ~0.65                       |
| Validation Strategy | Stratified Cross Validation |

### Key Achievements

* Built churn prediction system on 7K+ customers
* Trained multiple classification models
* Implemented threshold optimization
* Developed ensemble voting classifier
* Created reusable preprocessing pipeline
* Deployed prediction service using Flask

---

## Tech Stack

### Programming Languages

* Python
* SQL

### Machine Learning

* Scikit-learn
* Pandas
* NumPy

### Visualization

* Matplotlib

### Deployment

* Flask

### Tools

* Git
* GitHub
* VS Code

---

## Repository Structure

```text
customer-churn-prediction/
│
├── Data/
│   ├── telco_churn.csv
│
├── models/
│
├── notebooks/
│   └── churn_eda.ipynb
│
├── src/
│   ├── train_model.py
│   └── predict.py
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/geetha-akula/customer-churn-prediction.git
```

Navigate to project directory:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Flask application:

```bash
python app.py
```

The application will be available locally.

---

## Future Improvements

* Docker containerization
* Cloud deployment
* Model monitoring
* Automated retraining pipeline
* Advanced feature engineering
* CI/CD integration

---

## Author

**Akula Geetha Maheswari**

IIT Madras Data Science Graduate

LinkedIn:
https://linkedin.com/in/geetha-akula

GitHub:
https://github.com/geetha-akula

Email:
[geethaakula123@gmail.com](mailto:geethaakula123@gmail.com)

---

⭐ If you found this project useful, feel free to star the repository.
