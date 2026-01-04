from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Predicts on the test set and calculates performance metrics.
    
    Metrics:
    - MSE (Mean Squared Error): Average squared difference (lower is better).
    - RMSE (Root MSE): Average error in the same units as the grade (0-20)(lower is better).
    - R2 Score: How much variance the model explains (1.0 is perfect, 0.0 is bad).
    """
    # 1. Predict
    y_pred = model.predict(X_test)
    
    # 2. Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 3. Print Results
    print(f"\n-> {model_name} Evaluation ")
    print(f"RMSE (Average Error): {rmse:.2f} points")
    print(f"R2 Score (Accuracy):  {r2:.4f}")
    
    return {"rmse": rmse, "r2": r2}
