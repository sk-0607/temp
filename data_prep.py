import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def fix_missing_values(df):
    """
    Check and fix missing values in the key features.
    Uses median imputation for missing values if any.
    """
    features = ['Time', 'Amount']
    for feature in features:
        if df[feature].isnull().sum() > 0:
            median_val = df[feature].median()
            df[feature].fillna(median_val, inplace=True)
            print(f"Missing values in '{feature}' fixed using median: {median_val}")
        else:
            print(f"No missing values found in '{feature}'.")
    return df

def scale_features(df):
    """
    Scale the 'Time' and 'Amount' features to a [0, 1] range using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    print("Scaled 'Time' and 'Amount' to the range [0, 1].")
    return df

def balance_dataset(df):
    """
    Balance the dataset by undersampling the normal transactions (Class = 0)
    to match the number of fraud cases (Class = 1).
    """
    fraud_df = df[df['Class'] == 1]
    normal_df = df[df['Class'] == 0]
    
    fraud_count = len(fraud_df)
    # Randomly sample normal transactions to match the number of fraud cases
    normal_df_balanced = normal_df.sample(n=fraud_count, random_state=42)
    
    # Concatenate the fraud and balanced normal transactions
    balanced_df = pd.concat([fraud_df, normal_df_balanced])
    print(f"Balanced dataset: {fraud_count} fraud cases and {fraud_count} normal transactions.")
    return balanced_df

def preprocess_data(input_filepath, output_filepath):
    """
    Complete preprocessing pipeline:
    - Load data.
    - Fix missing values.
    - Scale key features ('Time' and 'Amount').
    - Retain key features and class label.
    - Balance the dataset.
    - Shuffle and save the preprocessed data.
    """
    df = load_data(input_filepath)
    if df is None:
        return
    
    # Fix missing values in key features
    df = fix_missing_values(df)
    
    # Scale 'Time' and 'Amount'
    df = scale_features(df)
    
    # Select only key features and the target label
    df = df[['Time', 'Amount', 'Class']]
    
    # Balance the dataset
    df = balance_dataset(df)
    
    # Shuffle the data to mix fraud and normal cases
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the preprocessed dataset
    df.to_csv(output_filepath, index=False)
    print(f"Preprocessed data saved to {output_filepath}")

def main():
    input_filepath = "creditcard.csv"              # Ensure this file is in your working directory.
    output_filepath = "creditcard_preprocessed.csv"
    preprocess_data(input_filepath, output_filepath)
    

if __name__ == "__main__":
    main()
