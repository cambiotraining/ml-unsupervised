# code to teach in Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# where are we?
print( os.getcwd() )

# change directory to where the data is stored
os.chdir('course_files/data/')

# where are we now?
print( os.getcwd() )

# load the diabetes data
df_diabetes = pd.read_csv('diabetes_sample_data.csv')
print( df_diabetes.head() )