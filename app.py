import gradio as gr
import torch
import json
import numpy as np
from training_model import RecipeAutoencoder, Config
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parse_parquet import load_parquet_file
import requests
from PIL import Image
from io import BytesIO

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

        # Load recipe data from Parquet using the parser
        self.df = load_parquet_file()
        if self.df is None:
            raise RuntimeError("Failed to load recipe data from Parquet file")
        
        # Create vectorizer with loaded vocabulary
        self.vectorizer = CountVectorizer(vocabulary=self.vocab)
        
        # Pre-compute all recipe embeddings
        self.recipe_embeddings = self._compute_all_embeddings()

    def _compute_all_embeddings(self):
        print("Computing recipe embeddings...")
        # Convert numpy arrays to strings and clean them
        all_ingredients = self.df['RecipeIngredientParts'].apply(lambda x: ', '.join(x.astype(str))).str.lower()
        ingredient_matrix = self.vectorizer.transform(all_ingredients)
        
        # Find max value for normalization (using sparse matrix)
        max_val = ingredient_matrix.max()
        if max_val == 0:
            max_val = 1  # Prevent division by zero
        
        embeddings = []
        batch_size = 32  # Reduced batch size
        
        print("Computing embeddings in batches...")
        with torch.no_grad():
            for i in range(0, ingredient_matrix.shape[0], batch_size):
                # Process each batch in sparse format until the last moment
                batch = ingredient_matrix[i:i+batch_size]
                # Convert to dense and normalize only the current batch
                batch_dense = batch.toarray().astype(np.float32)  # Use float32 instead of float64
                batch_dense = np.clip(batch_dense / max_val, 0, 1)
                
                batch_tensor = torch.FloatTensor(batch_dense).to(Config.DEVICE)
                _, batch_embedding = self.model(batch_tensor)
                embeddings.append(batch_embedding.cpu().numpy())
                
                # Clear some memory
                del batch_dense
                del batch_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return np.vstack(embeddings)

    def find_similar_recipes(self, ingredients: str, num_recipes: int = 5) -> list:
        """
        Find recipes similar to the given ingredients.

        Args:
            ingredients (str): Comma-separated list of ingredients
            num_recipes (int): Number of recipes to return

        Returns:
            list: List of recipe dictionaries containing name, ingredients, quantities, instructions, and image
        """
        # Convert string to list and clean
        ingredient_list = [ing.strip().lower() for ing in ingredients.split(',')]
        
        # Get embedding for input ingredients
        ingredients_str = ', '.join(ingredient_list)
        vec = self.vectorizer.transform([ingredients_str])
        
        # Normalize the vector the same way as during training
        max_val = vec.max()
        if max_val == 0:
            max_val = 1  # Prevent division by zero
            
        # Convert to dense and normalize
        vec_array = vec.toarray().astype(np.float32)  # Use float32 instead of float64
        vec_array = np.clip(vec_array / max_val, 0, 1)
        
        with torch.no_grad():
            tensor = torch.FloatTensor(vec_array).to(Config.DEVICE)
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
                'ingredients': recipe['RecipeIngredientParts'],  # Already a list in Parquet
                'RecipeIngredientQuantities': recipe['RecipeIngredientQuantities'],  # Add quantities
                'instructions': recipe['RecipeInstructions'],  # Already a list in Parquet
                'similarity': float(similarities[0][idx]),
                'image': recipe['Images']  # Add image URL
            })
        
        return results

def download_image(url):
    """Download image from URL and return as PIL Image object."""
    try:
        response = requests.get(url, timeout=5)
        return Image.open(BytesIO(response.content))
    except:
        # Return a default image or None if download fails
        return None

# Initialize the recommender
try:
    recommender = RecipeRecommender()
except Exception as e:
    print(f"Error initializing recommender: {e}")
    raise

def get_recipe_recommendations(ingredients: str, num_recipes: int = 5):
    """
    Get recipe recommendations based on ingredients.
    
    Args:
        ingredients (str): Comma-separated list of ingredients
        num_recipes (int): Number of recipes to return
    
    Returns:
        tuple: (list of images, formatted text output)
    """
    recipes = recommender.find_similar_recipes(ingredients, num_recipes)
    
    # Prepare images and text output
    images = []
    output = []
    
    for i, recipe in enumerate(recipes, 1):
        # Handle image - check if it's a non-empty array
        image_data = recipe['image']
        if isinstance(image_data, (list, np.ndarray)) and len(image_data) > 0:
            # Try each URL in the array until we find one that works
            for img_url in image_data:
                if isinstance(img_url, str) and img_url.startswith('https://img.sndimg.com/'):
                    img = download_image(img_url)
                    if img:
                        images.append(img)
                        break  # Stop after first successful image
        
        # Format recipe text
        output.append(f"\n{i}. {recipe['name']} (Similarity: {recipe['similarity']:.2f})")
        
        # Format ingredients with quantities
        output.append("\nIngredients:")
        quantities = recipe['RecipeIngredientQuantities']
        ingredients = recipe['ingredients']
        
        # Ensure quantities and ingredients are lists
        if isinstance(quantities, str):
            quantities = eval(quantities)
        if isinstance(ingredients, str):
            ingredients = eval(ingredients)
            
        # Zip quantities and ingredients together
        for qty, ing in zip(quantities, ingredients):
            if qty and qty.strip():  # If quantity exists
                output.append(f"- {qty.strip()} {ing}")
            else:
                output.append(f"- {ing}")
        
        # Format instructions
        output.append("\nInstructions:")
        instructions = recipe['instructions']
        if isinstance(instructions, str):
            instructions = eval(instructions)
            
        for idx, step in enumerate(instructions, 1):
            output.append(f"{idx}. {step}")
        
        output.append("\n" + "="*50)  # Add separator between recipes
    
    # If no valid images were found, return an empty list
    return images or [], "\n".join(output)

# Create the Gradio interface
demo = gr.Interface(
    fn=get_recipe_recommendations,
    inputs=[
        gr.Textbox(label="Ingredients (comma-separated)", placeholder="chicken, rice, soy sauce"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Recipes")
    ],
    outputs=[
        gr.Gallery(label="Recipe Images"),
        gr.Textbox(label="Recipe Details", lines=25)
    ],
    title="AI Recipe Recommender",
    description="Enter ingredients to find similar recipes. Separate ingredients with commas.",
    examples=[
        ["chicken, rice, soy sauce", 3],
        ["pasta, tomato, garlic, basil", 3],
        ["chocolate, flour, sugar, eggs", 3],
        ["potato, onion, cheese", 3],
        ["eggs", 3]
    ]
)

# Launch with MCP server enabled
if __name__ == "__main__":
    demo.launch(mcp_server=True) 