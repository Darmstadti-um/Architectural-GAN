from ctgan import CTGAN
import pandas as pd
# Extract categorical data types
raw_df = pd.read_csv('/content/raw_df - Лист1.csv')
categoricals = raw_df.select_dtypes(exclude="number").columns.tolist()

# Fit CTGAN
ctgan = CTGAN(epochs=150)
ctgan.fit(raw_df, categoricals)

# Generate the data
synthetic_data = ctgan.sample(20000)
synthetic_data.head()
