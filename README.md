Customer Churn Prediction

Overview

This project builds a machine learning pipeline to predict whether a telecom customer is likely to churn. It includes data preprocessing, exploratory data analysis, class imbalance handling, model comparison, evaluation, and a reusable prediction pipeline.

The objective is to identify high-risk customers early so businesses can take proactive retention actions.

⸻

Dataset

The project uses the Telco Customer Churn Dataset, which contains customer demographics, service subscriptions, billing details, tenure, and churn labels.

Target variable
	•	Churn → 1 (customer leaves), 0 (customer stays)

⸻

Features & Preprocessing

Data preparation includes:
	•	Removing irrelevant identifier columns
	•	Cleaning missing values in TotalCharges
	•	Converting numeric fields to correct data types
	•	Label encoding categorical variables
	•	Saving encoders for reproducible inference

This ensures consistent preprocessing during both training and prediction.

⸻

Exploratory Data Analysis

EDA is performed to understand customer behavior and churn patterns:
	•	Distribution analysis using histograms
	•	Outlier detection using box plots
	•	Correlation heatmap for numerical features
	•	Categorical feature count visualization

These analyses guide feature understanding and model selection.

⸻

Class Imbalance Handling

Customer churn data is naturally imbalanced. To address this:
	•	SMOTE oversampling is applied to the training set

This improves model fairness and predictive reliability.

⸻

Model Training & Evaluation

Multiple classifiers are compared using cross-validation:
	•	Decision Tree
	•	Random Forest
	•	XGBoost

Random Forest is selected as the final model based on performance.

Evaluation metrics include:
	•	Accuracy
	•	Confusion matrix
	•	Precision, recall, and F1-score

⸻

Prediction Pipeline

The trained system supports inference on new customer data:
	1.	Load trained model
	2.	Load saved encoders
	3.	Encode categorical inputs
	4.	Generate churn prediction
	5.	Output probability score

This structure enables deployment or integration into applications.

⸻

Project Structure

customer-churn-prediction/
│
├── dataset/
│   └── telco_customer_churn.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── models/
│   ├── customer_churn_model.pkl
│   └── LabelEncoder.pkl
│
├── requirements.txt
└── README.md


⸻

Technologies Used
	•	Python
	•	NumPy & Pandas
	•	Matplotlib & Seaborn
	•	Scikit-learn
	•	XGBoost
	•	Imbalanced-learn
	•	Pickle

Installation:

Clone the repository and install dependencies

git clone <repository-url>
cd customer-churn-prediction
pip install -r requirements.txt

Usage:

Run the notebook to:
	•	Clean and explore data
	•	Train models
	•	Evaluate performance
	•	Save encoders and trained model


Example Output

The system predicts:
	•	Customer churn class (Yes / No)
	•	Associated prediction probability

⸻

Future Enhancements
	•	Hyperparameter tuning
	•	Feature importance analysis
	•	API/web deployment
	•	Automated training pipeline
