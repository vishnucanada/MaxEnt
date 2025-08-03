from basis import analysis
import pandas as pd
path = r'C:\Users\evgisar\OneDrive - Ericsson\Desktop\Princ. Max Entr\data\cars_data.csv'

df = pd.read_csv(path).copy()

for col in df.columns:
    if df[col].dtype == 'object':
        continue
    else:
        analysis(df[col].head(1000).dropna())

