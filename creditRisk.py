import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dataPreprocessing import load_data, preprocess_data
from model import train_model


def main():
    st.title("Credit Risk Prediction")
    st.write("Predict the credit risk of loan applicants using machine learning.")

    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)

    # Train the model
    model, X_test, y_test, y_pred, y_proba, cv_scores = train_model(X, y)

    # Display results
    st.subheader("Model Evaluation")

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    # Cross-Validation Results
    st.subheader("Cross-Validation Results")
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Risky', 'Risky'], yticklabels=['Not Risky', 'Risky'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

    # AUC-ROC Curve
    st.subheader("AUC-ROC Curve")
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.Series(model.coef_[0], index=X.columns)
    feature_importance = feature_importance.sort_values(key=abs, ascending=False)

    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance (Logistic Regression Coefficients)")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    st.pyplot(fig)

    # Insights and Recommendations
    st.subheader("Insights and Recommendations")
    st.write("""
    - **Savings to Credit Ratio**: Applicants with a higher savings-to-credit ratio are less risky. Encourage applicants to maintain a healthy savings-to-credit ratio.
    - **Debt-to-Income Ratio**: Applicants with a high debt-to-income ratio are riskier. Consider limiting loan amounts for such applicants.
    - **Credit Duration Ratio**: Applicants with a high credit duration ratio are riskier. Consider reducing loan durations for such applicants.
    - **Housing Stability**: Applicants who own their homes are less risky. Consider offering better loan terms to such applicants.
    - **Job Level**: Skilled or highly skilled applicants (Job >= 2) are less risky. Focus on these applicants for lower interest rates.
    - **Purpose of Loan**: Applicants with essential purposes like 'car' or 'business' are less risky. Implement stricter policies for non-essential purposes like 'vacation' or 'radio/TV'.
    """)

if __name__ == "__main__":
    main()