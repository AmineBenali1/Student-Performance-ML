import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style for nicer plots
sns.set(style="whitegrid")

def create_plots(df, output_dir="results/plots"):
    """
    Generates and saves exploratory data analysis plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("Generating plots...")

    # 1. Target Distribution (G3)
    # What it tells us: Is the data balanced? Are grades mostly high, low, or average?
    plt.figure(figsize=(8, 6))
    sns.histplot(df['G3'], kde=True, bins=20, color='blue')
    plt.title('Distribution of Final Grades (G3)')
    plt.xlabel('Final Grade (0-20)')
    plt.ylabel('Count of Students')
    plt.savefig(f"{output_dir}/grade_distribution.png")
    plt.close()
    print(f"- Saved: {output_dir}/grade_distribution.png")

    # 2. Correlation Heatmap (Numerical features only)
    # What it tells us: Which features are strongly related? (e.g., G1 vs G3)
    # 1.0 means perfect positive correlation, -1.0 means perfect negative.
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    # Compute correlation matrix
    corr = numeric_df.corr()
    
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    print(f"- Saved: {output_dir}/correlation_heatmap.png")

    # 3. Study Time vs Grade (Box Plot)
    # What it tells us: Do students who study more get higher grades?
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='studytime', y='G3', data=df)
    plt.title('Study Time vs Final Grade')
    plt.xlabel('Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)')
    plt.ylabel('Final Grade (G3)')
    plt.savefig(f"{output_dir}/studytime_vs_grade.png")
    plt.close()
    print(f"- Saved: {output_dir}/studytime_vs_grade.png")

    # 4. Failures vs Grade (Scatter/Jitter Plot)
    # What it tells us: How severely do past failures impact the final grade?
    plt.figure(figsize=(8, 6))
    sns.stripplot(x='failures', y='G3', data=df, jitter=True, alpha=0.5)
    plt.title('Past Failures vs Final Grade')
    plt.xlabel('Number of Past Class Failures')
    plt.ylabel('Final Grade (G3)')
    plt.savefig(f"{output_dir}/failures_vs_grade.png")
    plt.close()
    print(f"- Saved: {output_dir}/failures_vs_grade.png")

if __name__ == "__main__":
    from data_loader import load_data
    
    data_path = "data/student-mat.csv"
    if not os.path.exists("data") and os.path.exists("../data"):
        data_path = "../data/student-mat.csv"
        
    try:
        df = load_data(data_path)
        create_plots(df)
        print("\nEDA completed! Check the 'results/plots' folder.")
    except Exception as e:
        print(f"Error: {e}")
