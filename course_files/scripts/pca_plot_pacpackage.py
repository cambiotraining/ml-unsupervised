# pip install pca pandas

from pca import pca
import pandas as pd

# Load the USArrests data
url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/USArrests.csv"
df = pd.read_csv(url, index_col=0)

# Use state names as labels
labels = df.index.tolist()

# Run PCA and plot
model = pca(n_components=2)
_ = model.fit_transform(df, row_labels=labels)
model.scatter(label=True, legend=False)  # set legend=True if you also pass group labels