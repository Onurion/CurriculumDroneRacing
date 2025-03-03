import os
import re
from collections import defaultdict

# Path to the main folder
main_folder = "tuning"

# Dictionary to store folder names and their corresponding values
results = defaultdict(float)

# Iterate through all folders in the main directory
for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        param_file = os.path.join(folder_path, "parameters.txt")

        # Check if parameters.txt exists
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                content = f.read()
                # Search for Mean Targets Reached value
                match = re.search(r'Mean Targets Reached:\s*([-+]?\d*\.?\d+)', content)
                if match:
                    value = float(match.group(1))
                    results[folder] = value

# Sort folders by value in descending order
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

# Print results
print("\nFolders sorted by Mean Targets Reached (highest to lowest):")
print("-" * 50)
print("Folder Name".ljust(30) + "Mean Targets Reached")
print("-" * 50)
for folder, value in sorted_results:
    print(f"{folder:<30} {value:.2f}")