import sys
from src.data_loader import load_data
from src.preprocess import preprocess_data, split_data
from src.train import train_linear_regression, train_random_forest, save_model
from src.evaluate import evaluate_model
import os

def main():
    print("=== Student Performance Prediction Pipeline ===\n")
    
    # 1. Load Data
    data_path = "data/student-mat.csv"
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Running download logic...")
        # Fallback or exit
        return

    df = load_data(data_path)
    if df is None:
        return
        
    # 2. Preprocess
    print("\n--- Preprocessing ---")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training Data: {X_train.shape[0]} students")
    print(f"Testing Data:  {X_test.shape[0]} students")
    
    # 3. Train & Evaluate Linear Regression
    print("\n--- Model 1: Linear Regression ---")
    lr_model = train_linear_regression(X_train, y_train)
    save_model(lr_model, "linear_regression.pkl")
    evaluate_model(lr_model, X_test, y_test, model_name="Linear Regression")
    
    # 4. Train & Evaluate Random Forest
    print("\n--- Model 2: Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, "random_forest.pkl")
    evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
    
    # 5. Show Prediction Examples
    print("\n--- Comparison: Actual grade vs Predicted grade ---")
    import pandas as pd
    
    # Get predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Create comparison table (First 10 students)
    comparison = pd.DataFrame({
        'Actual': y_test,
        'LinearReg': lr_pred,
        'RandomForest': rf_pred
    })
    
    comparison.index.name = "ID"
    # Reset index so 'ID' becomes a regular column and prints on the same line
    print(comparison.head(10).reset_index().round(1).to_string(index=False))

    # 6. Save Visualization 
    print("\n--- Generating Visualization ---")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, lr_pred, color='blue', alpha=0.5)
    plt.plot([0, 20], [0, 20], color='red', linestyle='--')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    
    # Plot 2: Random Forest
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, rf_pred, color='green', alpha=0.5)
    plt.plot([0, 20], [0, 20], color='red', linestyle='--')
    plt.title('Random Forest: Actual vs Predicted')
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    
    plt.tight_layout()
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig("results/model_comparison.png", dpi=300)
    print("Saved plot to: results/model_comparison.png")

if __name__ == "__main__":
    main()
