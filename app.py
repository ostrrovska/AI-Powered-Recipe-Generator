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

    def get_recipe_embedding(self, ingredients):
        # Convert ingredients to lowercase and clean
        ingredients = [ing.strip().lower() for ing in ingredients]
        # Create ingredient string
        ingredient_str = ', '.join(ingredients)
        # Vectorize
        vec = self.vectorizer.transform([ingredient_str])
        
        # Get embedding
        with torch.no_grad():
            tensor = torch.FloatTensor(vec.toarray()).to(Config.DEVICE)
            _, embedding = self.model(tensor)
        
        return embedding.cpu().numpy()

    def find_similar_recipes(self, ingredients, n=5):
        """Find similar recipes based on ingredients"""
        # Get embedding for input ingredients
        query_embedding = self.get_recipe_embedding(ingredients)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.recipe_embeddings)
        
        # Get top N similar recipes
        top_indices = similarities[0].argsort()[-n:][::-1]
        
        results = []
        for idx in top_indices:
            recipe = self.df.iloc[idx]
            results.append({
                'name': recipe['Name'],
                'ingredients': recipe['RecipeIngredientParts'],
                'instructions': recipe['RecipeInstructions'],
                'similarity': similarities[0][idx]
            })
        
        return results

def main():
    # Initialize recommender
    print("Initializing recipe recommender...")
    recommender = RecipeRecommender()
    
    # Test cases
    test_ingredients = [
        ["chicken", "rice", "soy sauce"],
        ["pasta", "tomato", "garlic", "basil"],
        ["chocolate", "flour", "sugar", "eggs"],
        ["potato", "onion", "cheese"]
    ]
    
    # Try each test case
    for ingredients in test_ingredients:
        print("\n" + "="*50)
        print(f"\nFinding recipes similar to: {', '.join(ingredients)}")
        similar_recipes = recommender.find_similar_recipes(ingredients, n=3)
        
        for i, recipe in enumerate(similar_recipes, 1):
            print(f"\n{i}. {recipe['name']} (Similarity: {recipe['similarity']:.2f})")
            print(f"Ingredients: {recipe['ingredients']}")
            print(f"Instructions: {recipe['instructions'][:200]}...")

if __name__ == "__main__":
    main() 