import os
import numpy as np
from datetime import datetime

# Define the directory path containing subfolders and the time range
base_dir = "/media/kc401/diskA/CQ/deeplearning/MVRADAR/save_model/all/alignment/attention"  # Replace with your actual directory
start_time = datetime.strptime("2024-12-12_13-28-04", "%Y-%m-%d_%H-%M-%S")  # Replace with desired start time
end_time = datetime.strptime("2024-12-12_14-02-20", "%Y-%m-%d_%H-%M-%S")  # Replace with desired end time

# Initialize lists to store per-class accuracies
all_accuracies = []

# Iterate through folders in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Check if the folder name matches the time format and is within the time range
    try:
        folder_time = datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        continue  # Skip non-matching folder names

    if not (start_time <= folder_time <= end_time):
        continue

    # Search for confusion_matrix.txt within the folder
    confusion_matrix_path = os.path.join(folder_path, 'output',"confusion_matrix.txt")
    if os.path.isfile(confusion_matrix_path):
        # Read confusion_matrix.txt and calculate accuracies
        with open(confusion_matrix_path, 'r') as f:
            matrix = np.loadtxt(f)

        # Calculate accuracy for each class
        per_class_accuracies = matrix.diagonal() / matrix.sum(axis=1)
        all_accuracies.append(per_class_accuracies)

# Convert to NumPy array for easier statistics computation
if all_accuracies:
    all_accuracies = np.array(all_accuracies)
    mean_accuracies = all_accuracies.mean(axis=0)
    variance_accuracies = all_accuracies.var(axis=0)

    # Print results
    print("Per-class Mean Accuracies:", mean_accuracies)
    print("Per-class Variance Accuracies:", variance_accuracies)
else:
    print("No confusion_matrix.txt files found in the specified time range.")
