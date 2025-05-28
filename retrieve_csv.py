import pandas as pd

def load_recipes():
    return pd.read_csv('db/recipes.csv')

recipes = load_recipes()
recipes = recipes.dropna(subset=['RecipeIngredientQuantities'])

print(recipes.head())
print(recipes.info())
print(recipes.describe())

print(recipes['Name'].isnull().sum())

print(recipes['RecipeIngredientParts'].isnull().sum())

print(recipes['RecipeIngredientQuantities'].isnull().sum()) 

print(recipes['RecipeInstructions'].isnull().sum())

print(recipes[['Name', 'RecipeIngredientParts', 'RecipeIngredientQuantities', 'RecipeInstructions']].head())


