"""
Customer Churn Prediction 🔮
-----------------------------
An enhanced Customer Churn Prediction project using XGBoost and Streamlit.
This script saves the scaler and feature columns to ensure full compatibility
between the training and prediction phases.

Execution:
1️⃣ Generate synthetic data (once):
    python customer_churn_prediction.py --generate

2️⃣ Train the model:
    python customer_churn_prediction.py --train

3️⃣ Run the Streamlit interface:
    streamlit run customer_churn_prediction.py
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from io import StringIO

# --- 1. Constants and File Settings ---
DATA_FILE = 'customer_data.csv'
MODEL_FILE = 'churn_model.joblib'
SCALER_FILE = 'scaler.joblib'
COLUMNS_FILE = 'feature_columns.joblib'
TARGET_COLUMN = 'Churn'

# --- 2. Synthetic Data Generation Function ---

def generate_synthetic_data(num_records=5000):
    """
    Generates synthetic data for customer churn prediction.
    """
    print(f"🔎 Generating {num_records} synthetic data records...")

    # Create customer feature data
    data = {}
    data['Gender'] = np.random.choice(['Male', 'Female'], num_records)
    data['SeniorCitizen'] = np.random.choice([0, 1], num_records, p=[0.85, 0.15])
    data['Partner'] = np.random.choice(['Yes', 'No'], num_records)
    data['Dependents'] = np.random.choice(['Yes', 'No'], num_records)
    data['tenure'] = np.random.randint(1, 73, num_records) # Tenure in months
    data['PhoneService'] = np.random.choice(['Yes', 'No'], num_records)
    data['MultipleLines'] = np.random.choice(['Yes', 'No', 'No phone service'], num_records)
    data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], num_records)
    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_records, p=[0.55, 0.25, 0.20])
    data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], num_records)
    data['MonthlyCharges'] = np.random.uniform(18.25, 120.0, num_records).round(2)
    # Calculate TotalCharges based on monthly charges and tenure with some variance
    data['TotalCharges'] = data['MonthlyCharges'] * data['tenure'] * np.random.uniform(0.9, 1.1, num_records).round(2)
    
    # Merge data into a DataFrame
    df = pd.DataFrame(data)

    # Add the target variable (Churn) based on weighted probabilities
    # Month-to-month contract, Fiber optic, and Senior Citizens are weighted higher for churn
    churn_prob = (
        0.10 +  # Base probability
        0.15 * (df['SeniorCitizen'] == 1) +
        0.25 * (df['Contract'] == 'Month-to-month') +
        0.20 * (df['InternetService'] == 'Fiber optic') +
        -0.10 * (df['tenure'] > 50) 
    ).clip(0.05, 0.95)

    df['Churn'] = np.random.rand(num_records) < churn_prob
    df['Churn'] = df['Churn'].map({True: 1, False: 0})

    # Save the file
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Data successfully generated and saved to: {DATA_FILE}")

# --- 3. Training and Evaluation Function ---

def preprocess_data(df, scaler=None, fit_scaler=True, columns_to_use=None):
    """
    Preprocesses the data for the model.
    Includes: handling missing values, one-hot encoding, and scaling.
    """
    
    # Convert TotalCharges to numeric, coercing errors to NaN (for real-world data)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values with the mean of TotalCharges
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # Identify column types
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove the target column from numerical features list
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)
        
    # One-Hot Encoding for categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Determine the final features after encoding
    feature_cols = numerical_cols + [col for col in df_encoded.columns if col not in df.columns or col in numerical_cols]
    
    # Extract the feature matrix
    X = df_encoded[feature_cols]

    # Ensure consistent columns during prediction/inference
    if columns_to_use is not None:
        # Add missing columns (if the input sample is missing a category)
        missing_cols = set(columns_to_use) - set(X.columns)
        for c in missing_cols:
            X[c] = 0
        # Reorder and filter columns to match the training data
        X = X[columns_to_use]
        feature_cols = columns_to_use

    # Scaling numerical columns (StandardScaler)
    if fit_scaler:
        # Fit and transform (during training)
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    elif scaler is not None:
        # Only transform (during prediction)
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    # Return processed features, target, scaler object, and final feature names
    return X, df_encoded[TARGET_COLUMN], scaler, feature_cols

def train_model():
    """
    Loads data, preprocesses it, trains the XGBoost model, and saves all necessary assets.
    """
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Data file {DATA_FILE} not found. Please run: 'python customer_churn_prediction.py --generate' first.")
        return

    print(f"⚙️ Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # Preprocess data and fit scaler
    X, y, scaler, feature_cols = preprocess_data(df, fit_scaler=True)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"✨ Training XGBoost model...")
    # Initialize the XGBoost model
    model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluation on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Evaluation Report on Test Data ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("------------------------------------------")

    # Save the model and its assets using joblib
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, COLUMNS_FILE)
    
    print(f"\n✅ Model saved to: {MODEL_FILE}")
    print(f"✅ Scaler saved to: {SCALER_FILE}")
    print(f"✅ Feature columns saved to: {COLUMNS_FILE}")


# --- 4. Streamlit Interface for Prediction (Main App) ---

@st.cache_resource
def load_assets():
    """Load the model, scaler, and feature columns from disk."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        feature_cols = joblib.load(COLUMNS_FILE)
        return model, scaler, feature_cols
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model files not found. Please run 'python customer_churn_prediction.py --train' first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
        st.stop()

def predict_churn(data, model, scaler, feature_cols):
    """
    Performs preprocessing and prediction on user input data.
    """
    
    # Convert input data dictionary to a DataFrame
    df = pd.DataFrame([data])
    
    # Convert numerical columns to appropriate type (Streamlit inputs might be strings)
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass 
            
    # Preprocess the data using the trained scaler (fit_scaler=False)
    X_processed, _, _, _ = preprocess_data(df, scaler=scaler, fit_scaler=False, columns_to_use=feature_cols)
    
    # Get the prediction probability (class 1: Churn)
    prediction_proba = model.predict_proba(X_processed)[:, 1][0]
    
    return prediction_proba

def streamlit_app():
    """
    The main Streamlit interface function.
    """
    st.set_page_config(page_title="Churn Prediction 🔮", layout="wide")
    
    st.title("🔮 Customer Churn Prediction System")
    st.markdown("A trained **XGBoost** model to predict the probability of a customer leaving the service.")
    
    # Load the model artifacts
    model, scaler, feature_cols = load_assets()

    # --- User Input Interface ---
    
    st.sidebar.header("📝 Customer Data")
    st.sidebar.markdown("Enter customer information for churn prediction:")

    # Input widgets for categorical features
    gender = st.sidebar.radio("Gender", ['Female', 'Male'])
    senior_citizen = st.sidebar.radio("Senior Citizen (65+)", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    partner = st.sidebar.radio("Partner", ['Yes', 'No'])
    dependents = st.sidebar.radio("Dependents", ['Yes', 'No'])
    phone_service = st.sidebar.radio("Phone Service", ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.radio("Paperless Billing", ['Yes', 'No'])

    # Input widgets for numerical features
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=70.0, step=0.5)
    total_charges = st.sidebar.number_input("Total Charges Paid ($)", min_value=18.0, max_value=8600.0, value=70.0 * tenure, step=1.0)


    # Aggregate input data into a dictionary
    input_data = {
        'Gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # --- Prediction and Results Display ---
    
    if st.sidebar.button("Predict Churn Probability"):
        with st.spinner('Performing prediction...'):
            churn_probability = predict_churn(input_data, model, scaler, feature_cols)
        
        churn_percentage = churn_probability * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Main Prediction Result")
            
            # Determine the risk level based on probability
            if churn_percentage >= 50:
                result_text = f"🚨 **High Probability of Churn!**"
                emoji = "🔴"
                color = "red"
            elif churn_percentage >= 30:
                result_text = f"⚠️ **Moderate Probability of Churn.**"
                emoji = "🟠"
                color = "orange"
            else:
                result_text = f"✅ **Low Probability of Churn.**"
                emoji = "🟢"
                color = "green"

            st.markdown(f"## {emoji} {result_text}")
            
            # Display the probability metric and progress bar
            st.metric(label="Customer Churn Probability", value=f"{churn_percentage:.2f}%")
            st.progress(churn_probability)
            
            st.markdown(f"> **Threshold:** If the probability is above **50%**, the customer is predicted to churn.")
            
        with col2:
            st.subheader("💡 Key Factor Analysis")
            
            # Highlight key features that contribute to churn risk
            st.markdown("* **Contract Type:** **Month-to-month** contracts significantly increase churn risk.")
            if input_data['InternetService'] == 'Fiber optic':
                st.markdown("* **Internet Service:** **Fiber optic** often correlates with higher churn rates due to competition or service issues.")
            if input_data['SeniorCitizen'] == 1:
                st.markdown("* **Age Group:** **Senior Citizens** sometimes show higher churn rates.")
            if input_data['tenure'] < 12:
                st.markdown(f"* **Tenure:** Short tenure (**{tenure} months**) indicates a newer customer who might leave early.")

# --- 5. Main Execution Block ---

def main():
    """
    Main function to handle Command Line Interface (CLI) arguments.
    """
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Project.")
    # Add arguments: --generate and --train
    parser.add_argument('--generate', action='store_true', help='Generate synthetic customer data and save to CSV.')
    parser.add_argument('--train', action='store_true', help='Train the XGBoost model and save assets.')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_synthetic_data()
    elif args.train:
        train_model()
    elif not args.generate and not args.train:
        # If no argument is passed, inform the user about the Streamlit command
        print("\n--- Streamlit Interface ---")
        print("To run the web app, use the command:")
        print("streamlit run customer_churn_prediction.py")
        print("------------------------")
        
        # Test loading assets if running outside the Streamlit environment
        try:
            load_assets()
            print("✅ Model and Scaler loaded successfully. Ready for Streamlit.")
        except:
            print("❌ Please run --train to create model files first.")
                
        
if __name__ == '__main__':
    # Environment variable check to determine if the script is run by Streamlit
    if 'streamlit' in os.environ.get('PYTHONPATH', ''):
        os.environ['RUNNING_STREAMLIT'] = 'True'
        # Streamlit execution calls the app function directly
        streamlit_app()
    else:
        main()