import os
import urllib.request
import zipfile

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "student.zip")
CSV_PATH = os.path.join(DATA_DIR, "student-mat.csv")

def download_data():
    # 1. Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # 2. Check if CSV file already exists
    if not os.path.exists(CSV_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
            print("Download complete.")

            print("Extracting dataset...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extract("student-mat.csv", DATA_DIR)
                zip_ref.extract("student-por.csv", DATA_DIR)
            
            # Clean up zip file
            os.remove(ZIP_PATH)
            print("Extraction complete. Zip file removed.")
        except Exception as e:
            print(f"Error downloading data: {e}")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_data()
