# Credit Risk Prediction

This project predicts the credit risk of loan applicants using machine learning. It leverages logistic regression and feature engineering to classify applicants as "Risky" or "Not Risky" based on their financial and demographic data.

---

## Features

- **Feature Engineering**:
  - Debt-to-Income Ratio
  - Credit Duration Ratio
  - Savings-to-Credit Ratio
  - Log Transformation of Credit Amount
  - Interaction Features
  - Age Group Binning
- **Model**:
  - Logistic Regression with hyperparameter tuning and cross-validation.
- **Insights**:
  - Feature importance visualization.
  - Actionable recommendations for improving credit risk evaluation.
- **Interactive UI**:
  - Built using Streamlit for easy interaction and visualization.

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Install the required libraries using `pip`.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Sarthak-Rakshe/CreditRiskPrediction
   cd CreditRiskPrediction

   ```

2. Install dependencies

   pip install -r requirements.txt

3.Run the application

    streamlit run creditRisk.py

#### Model Report

![Accuracy](<screenshots/Screenshot 2025-04-24 170324.png>)
![confusion matrix](<screenshots/Screenshot 2025-04-24 170445.png>)
![features](<screenshots/Screenshot 2025-04-24 170454.png>)
