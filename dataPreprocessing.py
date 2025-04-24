import pandas as pd
import numpy as np
from riskLogic import determine_credit_risk

def load_data():
    data = pd.read_csv(r'C:\Sarthak Rakshe\CreditRiskAnalysis\archive\german_credit_data.csv')

    # Convert 'Checking account' to numeric
    checking_account_mapping = {
        'little': 500, 'moderate': 2000, 'quite rich': 5000, 'rich': 8000, None: 0
    }
    data['Checking account'] = data['Checking account'].map(checking_account_mapping).fillna(0)

    # Convert 'Saving accounts' to numeric
    saving_accounts_mapping = {
        'little': 1000, 'moderate': 3000, 'quite rich': 7000, 'rich': 10000, None: 0
    }
    data['Saving accounts'] = data['Saving accounts'].map(saving_accounts_mapping).fillna(0)

    # Create a new feature: Savings to Credit Ratio
    data['Savings to Credit Ratio'] = data['Saving accounts'] / (data['Credit amount'] + 1)  # Avoid division by zero

    # Feature Engineering
    # 1. Debt-to-Income Ratio (using Checking account as a proxy for income)
    data['Debt-to-Income Ratio'] = data['Credit amount'] / (data['Checking account'] + 1)  # Avoid division by zero

    # 2. Credit Duration Ratio
    data['Credit Duration Ratio'] = data['Credit amount'] / (data['Duration'] + 1)  # Avoid division by zero

    # 3. Log Transformation of Credit Amount
    data['Log Credit Amount'] = np.log1p(data['Credit amount'])  # log1p handles log(0)

    # 4. Interaction Features
    data['Housing_Own_Credit'] = (data['Housing'] == 'own').astype(int) * data['Credit amount']

    # 5. Binning Age into Categories
    data['Age Group'] = pd.cut(data['Age'], bins=[18, 30, 45, 60, 100], labels=['Young', 'Adult', 'Middle-Aged', 'Senior'])

    # Apply the determine_credit_risk function to create the 'CreditRisk' column
    data['CreditRisk'] = data.apply(determine_credit_risk, axis=1)

    return data

def preprocess_data(data):
    # Encode categorical variables
    categorical_columns = ['Sex', 'Housing', 'Purpose', 'Age Group']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Feature-target split
    target_column = 'CreditRisk'
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y