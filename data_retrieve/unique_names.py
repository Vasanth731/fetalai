import os
import pandas as pd
from collections import Counter

# Path to the root "Patient" folder
root_dir = 'orig'

# Counter to hold image name frequencies
image_counter = Counter()

# Traverse the directory
for root, dirs, files in os.walk(root_dir):
    for file in files:
        image_counter[file] += 1

# Convert to DataFrame
df = pd.DataFrame(image_counter.items(), columns=['Image Name', 'Count'])

# Save to Excel
output_path = 'image_counts.csv'

df.to_csv(output_path, index=False)


print(f"Excel file saved to: {output_path}")
