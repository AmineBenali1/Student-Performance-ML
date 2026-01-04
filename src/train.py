from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_linear_regression(X_train, y_train):
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    # n_estimators=100 means we build 100 little trees
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    # Saves the trained model to the 'models/' directory.
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
