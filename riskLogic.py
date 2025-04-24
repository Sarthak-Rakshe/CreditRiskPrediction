def determine_credit_risk(row):
    # Risky Conditions
    # 1. Low savings-to-credit ratio (< 0.2) and high credit amount (> 12000)
    if row['Savings to Credit Ratio'] < 0.2 and row['Credit amount'] > 12000:
        return 1  # Risky

    # 2. High debt-to-income ratio (> 0.5) and high credit amount (> 10000)
    if row['Debt-to-Income Ratio'] > 0.5 and row['Credit amount'] > 10000:
        return 1  # Risky

    # 3. High credit duration ratio (> 500) and long loan duration (> 36 months)
    if row['Credit Duration Ratio'] > 500 and row['Duration'] > 36:
        return 1  # Risky

    # 4. High log-transformed credit amount (> 9) and unstable housing (rent or free)
    if row['Log Credit Amount'] > 9 and row['Housing'] in ['rent', 'free']:
        return 1  # Risky

    # 5. Low job level (0 or 1) and low savings-to-credit ratio (< 0.3)
    if row['Job'] in [0, 1] and row['Savings to Credit Ratio'] < 0.3:
        return 1  # Risky

    # 6. Very high credit amount (> 15000) and long loan duration (> 36 months)
    if row['Credit amount'] > 15000 and row['Duration'] > 36:
        return 1  # Risky

    # Not Risky Conditions
    # 1. High savings-to-credit ratio (>= 0.5) and stable housing (own)
    if row['Savings to Credit Ratio'] >= 0.5 and row['Housing'] == 'own':
        return 0  # Not Risky

    # 2. High job level (>= 2), low credit amount (<= 10000), and moderate savings-to-credit ratio (>= 0.3)
    if row['Job'] >= 2 and row['Credit amount'] <= 10000 and row['Savings to Credit Ratio'] >= 0.3:
        return 0  # Not Risky

    # 3. Essential loan purpose (car or business), low credit amount (<= 8000), and high savings-to-credit ratio (>= 0.4)
    if row['Purpose'] in ['car', 'business'] and row['Credit amount'] <= 8000 and row['Savings to Credit Ratio'] >= 0.4:
        return 0  # Not Risky

    # 4. Very high savings-to-credit ratio (>= 0.6) and short loan duration (<= 12 months)
    if row['Savings to Credit Ratio'] >= 0.6 and row['Duration'] <= 12:
        return 0  # Not Risky

    # 5. Low debt-to-income ratio (<= 0.3) and low credit duration ratio (<= 300)
    if row['Debt-to-Income Ratio'] <= 0.3 and row['Credit Duration Ratio'] <= 300:
        return 0  # Not Risky

    # Default fallback: If no conditions are met, classify as risky.
    return 1  # Risky