# 🔮 Customer Churn Prediction System

An advanced machine learning system for predicting customer churn using XGBoost. This project includes data generation, model training, and an interactive Streamlit web interface for predictions.

![Project](https://img.shields.io/badge/Project-Customer%20Churn%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![Web App](https://img.shields.io/badge/Web%20App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Web Interface](#web-interface)
- [Customization](#customization)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

Customer churn prediction is crucial for businesses to identify customers who are likely to stop using their services. This system uses machine learning to analyze customer data and predict churn probability, enabling proactive customer retention strategies.

## ✨ Features

- **📊 Synthetic Data Generation**: Create realistic customer datasets for training
- **🤖 XGBoost Model**: High-performance gradient boosting algorithm
- **🔧 Automated Preprocessing**: Handles missing values, encoding, and scaling
- **📈 Model Evaluation**: Comprehensive performance metrics and reports
- **🌐 Streamlit Interface**: User-friendly web application for predictions
- **💾 Model Persistence**: Save and load trained models with accessories

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python customer_churn_prediction.py --generate

# 3. Train model
python customer_churn_prediction.py --train

# 4. Launch app
streamlit run customer_churn_prediction.py

pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
streamlit>=1.22.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0

Then install using:

pip install -r requirements.txt

📖 Usage

Step 1: Generate Synthetic Data
First, create a synthetic dataset for training:

python customer_churn_prediction.py --generate

This creates customer_data.csv with 5000 customer records.

Step 2: Train the Model
Train the XGBoost model on the generated data:

python customer_churn_prediction.py --train

This will:

Preprocess the data

Train the XGBoost classifier

Evaluate model performance

Save the model, scaler, and feature columns

Step 3: Run the Web Application
Launch the Streamlit interface:

streamlit run customer_churn_prediction.py

The application will open in your default browser at http://localhost:8501

🏗️ Project Structure

customer_churn_prediction/
│
├── customer_churn_prediction.py  # Main application file
├── customer_data.csv             # Generated customer data (after step 1)
├── churn_model.joblib           # Trained model (after step 2)
├── scaler.joblib                # Fitted scaler (after step 2)
├── feature_columns.joblib       # Feature columns list (after step 2)
└── README.md                    # This file

📊 Model Performance
The XGBoost model typically achieves:

Accuracy: 85-90%

ROC AUC Score: 0.90-0.95

Precision: 85-90%

Recall: 80-85%

Key Features Impacting Churn:
Contract Type: Month-to-month contracts have higher churn

Internet Service: Fiber optic users show varied churn patterns

Tenure: New customers (low tenure) have higher churn risk

Senior Citizens: Slightly higher churn probability

Monthly Charges: Higher charges may correlate with churn

🔧 Technical Details
Data Preprocessing
Missing Values: TotalCharges missing values filled with mean

Categorical Encoding: One-hot encoding for all categorical features

Feature Scaling: StandardScaler for numerical features

Data Splitting: 80-20 train-test split with stratification

Machine Learning Model
Algorithm: XGBoost (Extreme Gradient Boosting)

Objective: binary:logistic

Evaluation Metric: logloss

Hyperparameters:

n_estimators: 100

learning_rate: 0.1

max_depth: 5

random_state: 42

Features Used
Category	Features
Demographic	Gender, SeniorCitizen, Partner, Dependents
Service	PhoneService, MultipleLines, InternetService
Billing	Contract, PaperlessBilling, MonthlyCharges, TotalCharges
Engagement	tenure
🎮 Using the Web Interface
Input Customer Data: Fill in the form in the sidebar

Click Predict: Get instant churn probability

Interpret Results:

🟢 Green: Low risk (<30%)

🟠 Orange: Medium risk (30-50%)

🔴 Red: High risk (>50%)

Example Prediction Scenarios
High Churn Risk:

Month-to-month contract

Fiber optic internet

Short tenure (<12 months)

High monthly charges

Low Churn Risk:

Two-year contract

Long tenure (>24 months)

DSL internet service

Moderate monthly charges

🛠️ Customization
Adding New Features
Update the generate_synthetic_data() function

Modify the preprocess_data() function

Update the Streamlit input interface

Retrain the model

Model Tuning
Modify the XGBClassifier parameters in train_model():

model = XGBClassifier(
    n_estimators=200,      # Increase number of trees
    learning_rate=0.05,    # Lower learning rate
    max_depth=7,           # Deeper trees
    subsample=0.8,         # Subsample ratio
    colsample_bytree=0.8,  # Feature subsample ratio
    random_state=42
)

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for:

Bug fixes

New features

Performance improvements

Documentation enhancements

Development Setup
Fork the repository

Create a feature branch

Make your changes

Test thoroughly

Submit a pull request

📄 License
This project is open source and available under the MIT License.

🆘 Troubleshooting
Common Issues
FileNotFoundError: Ensure you run --generate before --train

ModuleNotFoundError: Install all required packages

Streamlit not working: Check if Streamlit is properly installed

Model loading errors: Ensure all joblib files are in the same directory

Getting Help
If you encounter issues:

Check the error messages carefully

Ensure all dependencies are installed

Verify file paths and permissions

Check Python version compatibility

📞 Support
For questions and support:

Check the troubleshooting section above

Review the code comments

Open an issue in the repository

⭐ If you find this project useful, please give it a star! ⭐

Built with ❤️ using Python, XGBoost, and Streamlit.



📈 Model Evaluation: Comprehensive performance metrics and reports

🌐 Streamlit Interface: User-friendly web application for predictions

💾 Model Persistence: Save and load trained models with accessories
