import pandas as pd

__all__ = ['load_recipes', 'clean_null_values']

def load_recipes():
    return pd.read_csv('db/recipes.csv')

def clean_null_values(df, column_name):
    df = df.dropna(subset=[column_name])
    return df

df = load_recipes()
print(df.info())