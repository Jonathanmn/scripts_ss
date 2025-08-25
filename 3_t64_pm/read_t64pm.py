import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes



folder_t64 = 'C:/git_mn/scripts_ss/_files/pm/L0/minuto'

# Get all .txt files in the folder
txt_files = glob.glob(os.path.join(folder_t64, '*.txt'))

# Read all files and combine them
dataframes = []
for file in txt_files:
    df = pd.read_csv(file, sep=',', header=0)
    # Optional: add a column to identify the source file
    
    dataframes.append(df)

# Combine all dataframes
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Read {len(txt_files)} files with {len(combined_df)} total rows")
else:
    print("No .txt files found in the folder")



