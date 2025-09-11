# Load and Save NCI60 Cancer Data from GitHub
# This script fetches the NCI60 cancer dataset from a GitHub repository,
# processes it, and saves it as a CSV file.

import os
import pandas as pd
import numpy as np
import requests
import io
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# URLs for the raw data and labels from GitHub repository
data_url = 'https://raw.githubusercontent.com/neelsoumya/python_machine_learning/main/data/NCI60data.npy'
labs_url = 'https://raw.githubusercontent.com/neelsoumya/python_machine_learning/main/data/NCI60labs.csv'

# Load Data from GitHub
response = requests.get(data_url)
response.raise_for_status()
X = np.load(io.BytesIO(response.content))
print("NCI60 data loaded successfully from GitHub.")

# Load Labels from GitHub
print("Fetching labels from GitHub...")
response = requests.get(labs_url)
response.raise_for_status()
# Read the raw text and split into lines.
all_lines = response.text.strip().splitlines()

# Skip the first line (the header) to match the data dimensions.
labs = all_lines[1:]

# The labels in the file are quoted (e.g., "CNS"), so we remove the quotes.
labs = [label.strip('"') for label in labs]

# The data is in the variable named X
print(X[:5])
print(X.shape)

# Convert and save to pandas dataframe
row_names = [ f"sample_{i}" for i in range(X.shape[0]) ]
col_names =  [f"feature_{i}" for i in range(X.shape[1]) ]
df_X_saved = pd.DataFrame(data=X, index=row_names, columns=col_names)

df_X_saved.to_csv(path_or_buf="cancer_data_saved_NC160.csv")
print("Data saved to cancer_data_saved_NC160.csv")