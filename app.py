import gradio as gr
import torch
import json
import numpy as np
from training_model import RecipeAutoencoder, Config
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecipeRecommender:
    def __init__(self, model_path='best_recipe_model.pth', vocab_path='ingredient_vocab.json'):
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Initialize model
        self.model = RecipeAutoencoder(len(self.vocab))
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.DEVICE)
        self.model.eval()

        # Load recipe data
        self.df = pd.read_csv('db/recipes.csv')
        
        # Create vectorizer with loaded vocabulary
        self.vectorizer = CountVectorizer(vocabulary=self.vocab)
        
        # Pre-compute all recipe embeddings
        self.recipe_embeddings = self._compute_all_embeddings()

    def _compute_all_embeddings(self):
        print("Computing recipe embeddings...")
        all_ingredients = self.df['RecipeIngredientParts'].fillna('').str.lower()
        ingredient_matrix = self.vectorizer.transform(all_ingredients)
        
        embeddings = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, ingredient_matrix.shape[0], batch_size):
                batch = ingredient_matrix[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch.toarray()).to(Config.DEVICE)
                _, batch_embedding = self.model(batch_tensor)
                embeddings.append(batch_embedding.cpu().numpy())
        
        return np.vstack(embeddings)

    def find_similar_recipes(self, ingredients: str, num_recipes: int = 5) -> list:
        """
        Find recipes similar to the given ingredients.

        Args:
            ingredients (str): Comma-separated list of ingredients
            num_recipes (int): Number of recipes to return

        Returns:
            list: List of recipe dictionaries containing name, ingredients, and instructions
        """
        # Convert string to list
        ingredient_list = [ing.strip().lower() for ing in ingredients.split(',')]
        
        # Get embedding for input ingredients
        ingredients_str = ', '.join(ingredient_list)
        vec = self.vectorizer.transform([ingredients_str])
        
        with torch.no_grad():
            tensor = torch.FloatTensor(vec.toarray()).to(Config.DEVICE)
            _, embedding = self.model(tensor)
            query_embedding = embedding.cpu().numpy()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.recipe_embeddings)
        
        # Get top N similar recipes
        top_indices = similarities[0].argsort()[-num_recipes:][::-1]
        
        results = []
        for idx in top_indices:
            recipe = self.df.iloc[idx]
            results.append({
                'name': recipe['Name'],
                'ingredients': recipe['RecipeIngredientParts'],
                'instructions': recipe['RecipeInstructions'],
                'similarity': float(similarities[0][idx])
            })
        
        return results

# Initialize the recommender
recommender = RecipeRecommender()

def get_recipe_recommendations(ingredients: str, num_recipes: int = 5) -> str:
    """
    Get recipe recommendations based on ingredients.
    
    Args:
        ingredients (str): Comma-separated list of ingredients
        num_recipes (int): Number of recipes to return
    
    Returns:
        str: Formatted string containing recipe recommendations
    """
    recipes = recommender.find_similar_recipes(ingredients, num_recipes)
    
    # Format the output
    output = []
    for i, recipe in enumerate(recipes, 1):
        output.append(f"{i}. {recipe['name']} (Similarity: {recipe['similarity']:.2f})")
        output.append(f"Ingredients: {recipe['ingredients']}")
        output.append(f"Instructions: {recipe['instructions'][:200]}...")
        output.append("")
    
    return "\n".join(output)

# Create the Gradio interface
demo = gr.Interface(
    fn=get_recipe_recommendations,
    inputs=[
        gr.Textbox(label="Ingredients (comma-separated)", placeholder="chicken, rice, soy sauce"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Recipes")
    ],
    outputs=gr.Textbox(label="Recipe Recommendations"),
    title="AI Recipe Recommender",
    description="Enter ingredients to find similar recipes. Separate ingredients with commas.",
    examples=[
        ["chicken, rice, soy sauce", 3],
        ["pasta, tomato, garlic, basil", 3],
        ["chocolate, flour, sugar, eggs", 3],
        ["potato, onion, cheese", 3]
    ]
)

# Launch with MCP server enabled
if __name__ == "__main__":
    demo.launch(mcp_server=True) 