import polars as pl
import os
import time

folder_path = 'DATOS Sensores/pm/L0/minuto/'


df = pl.scan_csv(folder_path, has_header=True, separator=',').collect()

print(df)