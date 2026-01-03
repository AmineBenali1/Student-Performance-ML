import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
     prepares the data for training.
    
    1. Separate the Target (G3) from the Features (everything else).
    2. Convert categorical text features into numbers (One-Hot Encoding).
    """
    # 1. Separate Features (X) and Target (y)
    # Targe variable 'G3' is what we want to predict
    y = df['G3']
    
    # Drop 'G3' from the dataframe to get our features
    # Axis=1 (columns)
    X = df.drop(columns=['G3'])
    
    # 2. Handle Categorical Variables (One-Hot Encoding)
    # Example: 'sex' column has 'F' and 'M'. 
    # This function creates new columns: 'sex_M' (1 if Male, 0 if Female)
    # drop_first=True removes the first column to prevent redundancy
    # (e.g., if we know it's not Male, it must be Female)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Encoded features: {X_encoded.shape[1]} (Categorical variables expanded)")
    
    return X_encoded, y

def split_data(X, y):
    """
    Splits the data into Training and Testing sets.
    
    X_train, y_train: Data to teach the model (80%)
    X_test, y_test:   Data to test the model (20%)
    """
    # test_size=0.2 means 20% of data is for testing
    # random_state=42 ensures we get the same split every time we run this
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the code
    from data_loader import load_data
    import os
    
    # Handle path so we can run this file directly
    data_path = "data/student-mat.csv"
    if not os.path.exists("data") and os.path.exists("../data"):
        data_path = "../data/student-mat.csv"
        
    try:
        # Load
        df = load_data(data_path)
        
        # Preprocess
        print("\n--- Preprocessing ---")
        X, y = preprocess_data(df)
        
        # Split
        print("\n--- Splitting ---")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Testing shapes:  X={X_test.shape}, y={y_test.shape}")
        
        print("\nSuccess: Data is ready for machine learning!")
        
    except FileNotFoundError as e:
        print(e)
