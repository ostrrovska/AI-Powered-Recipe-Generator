import pandas as pd

def load_parquet_file(file_path='db/recipes.parquet'):
    """Load and return the Parquet file as a pandas DataFrame."""
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return None

def examine_parquet_structure():
    """Examine and print the structure of the Parquet file."""
    df = load_parquet_file()
    if df is not None:
        print("\nDataFrame Info:")
        print("="*50)
        print(df.info())
        
        print("\nFirst few rows:")
        print("="*50)
        print(df.head())
        
        print("\nColumn names:")
        print("="*50)
        print(df.columns.tolist())
        
        print("\nSample of RecipeIngredientParts:")
        print("="*50)
        if 'RecipeIngredientParts' in df.columns:
            print(df['RecipeIngredientParts'].head())
        else:
            print("RecipeIngredientParts column not found")

if __name__ == "__main__":
    examine_parquet_structure() 