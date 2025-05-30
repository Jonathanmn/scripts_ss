import pandas as pd
import time
'''
data = '/home/jmn/ss/scripts/git/scripts_ss/DATOS Sensores/pm/L0/2024_09_CMUL_PM.txt'

start = time.time()
df = pd.read_csv(data, header=0, sep=',')
end = time.time()

print(df.head())
print(f"Time to read DataFrame: {end - start:.4f} seconds")
'''






import polars as pl


param = '/home/jmn/ss/scripts/CMUL_tablero/scripts_graficas/CMUL_Minuto.dat'

# read CSV with Polars and measure time
start = time.time()
df = pl.read_csv(param)
end = time.time()

print(df.head())
print(f"Time to read DataFrame with Polars: {end - start:.4f} seconds")


