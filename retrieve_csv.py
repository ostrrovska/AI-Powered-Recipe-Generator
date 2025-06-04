import pandas as pd
import ast  

__all__ = ['load_recipes', 'clean_null_values', 'preprocess_ingredient_parts']

def load_recipes(path='db/recipes.csv'):
    return pd.read_csv(path)

def clean_null_values(df, column_name):
    return df.dropna(subset=[column_name])

def preprocess_ingredient_parts(df, column_name='RecipeIngredientParts'):
    def parse_ingredients(x):
        if not isinstance(x, str):
            return []
        # Remove c() wrapper and clean up quotes
        if x.startswith('c(') and x.endswith(')'):
            x = x[2:-1]
        # Split by commas and clean up quotes and whitespace
        ingredients = [ing.strip().strip('"\'') for ing in x.split(',')]
        return [ing for ing in ingredients if ing]

    df[column_name] = df[column_name].apply(parse_ingredients)
    return df

# Main code
if __name__ == '__main__':
    df = load_recipes()
    df = clean_null_values(df, 'RecipeIngredientParts')
    df = preprocess_ingredient_parts(df)

    print(df.info())
    print(df['RecipeIngredientParts'].head())
