import pandas as pd  # Import the pandas library for data manipulation
import os            # Import os to check if files exist

def load_data(file_path):
    """
    Loads the Student Performance dataset from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    print(f"Attempting to load data from: {file_path}")
    
    # Check if the file exists at the specified path
    if not os.path.exists(file_path):
        # If not, raise a FileNotFoundError with a helpful message
        raise FileNotFoundError(f"\nError: The file '{file_path}' was not found.\n"
                                f"Please make sure you have downloaded the dataset and placed it in the correct folder.")
    
    # Load the data
    # The dataset uses semicolons ';' as separators instead of commas
    try:
        df = pd.read_csv(file_path, sep=';')
        return df
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        return None

if __name__ == "__main__":
    # Define the default path to the dataset
    # This assumes you run the script from the project root folder
    default_path = "data/student-mat.csv"
    
    # Check if we are running inside the 'src' directory and adjust path if needed
    if not os.path.exists("data") and os.path.exists("../data"):
        default_path = "../data/student-mat.csv"

    try:
        # 1. Load the data
        student_data = load_data(default_path)
        
        if student_data is not None:
            # 2. Print dataset shape (rows, columns)
            print("\n--- Dataset Info ---")
            print(f"Shape: {student_data.shape} (Rows, Columns)")
            
            # 3. Print column names
            print("\n--- Columns ---")
            print(student_data.columns.tolist())
            
            # 4. Print first 5 rows to inspect data
            print("\n--- First 5 Rows ---")
            print(student_data.head())
            
            print("\nSuccess: Data loaded correctly!")
            
    except FileNotFoundError as e:
        print(e)
