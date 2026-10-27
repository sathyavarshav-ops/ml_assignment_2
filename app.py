import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="BITS ML Classifier Evaluator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Multi-Model ML Classification Dashboard")
st.markdown("Evaluate pre-trained machine learning models on custom uploaded test datasets.")

# Sidebar Controls
st.sidebar.header("🕹️ Controls & Configuration")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    )
)

model_files = {
    "Logistic Regression": "model/log_reg_model.pkl",
    "Decision Tree": "model/dt_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/nb_model.pkl",
    "Support Vector Machine": "model/svm_model.pkl"
}

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

@st.cache_resource
def load_artifacts(model_name):
    model = joblib.load(model_files[model_name])
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("📌 Uploaded Test Data Preview")
    st.dataframe(df_test.head(), use_container_width=True)
    
    if 'target' not in df_test.columns:
        st.error("Error: Uploaded dataset must contain a column named 'target'.")
    else:
        X_test = df_test.drop(columns=['target'])
        y_test = df_test['target']
        
        try:
            model, scaler = load_artifacts(model_choice)
            
            # Apply scaling for distance/gradient based models
            if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine", "Naive Bayes"]:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test.values
                
            y_pred = model.predict(X_eval)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_eval)[:, 1]
            else:
                y_proba = y_pred
                
            # Metrics Calculation
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.markdown("---")
            st.subheader(f"📈 Evaluation Metrics for **{model_choice}**")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Class 0', 'Class 1'],
                            yticklabels=['Class 0', 'Class 1'])
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
                
            with c2:
                st.subheader("📄 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose().style.highlight_max(axis=0))

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
else:
    st.info("👆 Please upload a valid `test_data.csv` file from the sidebar to view metrics.")