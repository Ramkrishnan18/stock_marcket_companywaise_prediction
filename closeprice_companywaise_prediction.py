import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Data path
CSV_PATH = r"C:\Users\HP\Downloads\archive (31)\stock_details_5_years.csv"  

# Load data 

def load_data():
    try:
        data = pd.read_csv(CSV_PATH)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def train_model(data):
    features = ['Open', 'High', 'Low', 'Volume', 'Company']
    target = 'Close'

    # Encode company name
    le = LabelEncoder()
    data['Company'] = le.fit_transform(data['Company'])

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'MSE_train': mean_squared_error(y_train, y_train_pred),
        'R2_train': r2_score(y_train, y_train_pred),
        'MAE_train': mean_absolute_error(y_train, y_train_pred),
        'MSE_test': mean_squared_error(y_test, y_test_pred),
        'R2_test': r2_score(y_test, y_test_pred),
        'MAE_test': mean_absolute_error(y_test, y_test_pred)
    }

    return model, le, metrics, y_test, y_test_pred

def save_model(model, le, metrics):
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

def load_model():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        return model, le, metrics
    except:
        return None, None, None


def save_prediction_history(input_data, prediction):
    history_file = 'prediction_history.csv'
    input_data['Predicted Close'] = prediction
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        history = pd.concat([history, pd.DataFrame([input_data])], ignore_index=True)
    else:
        history = pd.DataFrame([input_data])
    history.to_csv(history_file, index=False)

def clear_prediction_history():
    if os.path.exists('prediction_history.csv'):
        os.remove('prediction_history.csv')

# Streamlit App 

st.set_page_config(page_title="📈 Smart Stock Predictor", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "📊 Metrics", "🧾 History",  "🧹 Clear History"])

# Home 
if page == "🏠 Home":
    st.title("📈 Smart Stock Predictor (Company Aware)")
    st.markdown("""
        - Trains model using predefined CSV
        - Uses company name in prediction
        - View prediction, metrics, history, and more
    """)

    data = load_data()
    if data is not None:
        st.success("Data loaded successfully!")
        st.write("Sample Data:", data.head())
        model, le, metrics, y_test, y_pred = train_model(data)
        save_model(model, le, metrics)
        st.success("✅ Model trained and saved!")

# Manual Predict 
elif page == "📈 Predict":
    st.title("📈 Manual Stock Close Price Prediction")

    model, le, _ = load_model()
    if model is None:
        st.warning("Please go to Home and train the model first.")
    else:
        company_list = le.classes_
        company = st.selectbox("Select Company", company_list)
        open_val = st.number_input("Open", format="%.2f")
        high_val = st.number_input("High", format="%.2f")
        low_val = st.number_input("Low", format="%.2f")
        volume_val = st.number_input("Volume", format="%.2f")

        if st.button("Predict"):
            company_encoded = le.transform([company])[0]
            input_df = pd.DataFrame([{
                'Open': open_val,
                'High': high_val,
                'Low': low_val,
                'Volume': volume_val,
                'Company': company_encoded
            }])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Close Price: {prediction:.2f}")

            input_record = {
                'Company': company,
                'Open': open_val,
                'High': high_val,
                'Low': low_val,
                'Volume': volume_val
            }
            save_prediction_history(input_record, prediction)

# Metrics 
elif page == "📊 Metrics":
    st.title("📊 Model Performance Metrics")
    _, _, metrics = load_model()
    if metrics:
        st.metric("Train MSE", f"{metrics['MSE_train']:.2f}")
        st.metric("Train R2", f"{metrics['R2_train']:.2f}")
        st.metric("Train MAE", f"{metrics['MAE_train']:.2f}")
        st.metric("Test MSE", f"{metrics['MSE_test']:.2f}")
        st.metric("Test R2", f"{metrics['R2_test']:.2f}")
        st.metric("Test MAE", f"{metrics['MAE_test']:.2f}")
    else:
        st.warning("Metrics not found. Train the model first.")

# History 
elif page == "🧾 History":
    st.title("🧾 Prediction History")
    if os.path.exists('prediction_history.csv'):
        history = pd.read_csv('prediction_history.csv')
        st.dataframe(history)
    else:
        st.info("No prediction history available.")


# Clear History
elif page == "🧹 Clear History":
    st.title("🧹 Clear Prediction History")
    if st.button("Clear History"):
        clear_prediction_history()
        st.success("History cleared successfully!")