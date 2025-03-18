import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Завантажити дані
df = pd.read_csv("normalized_recipes.csv")

# Створити словник унікальних інгредієнтів
all_ingredients = list(set(ing for sublist in df["normalized_ingredients"] for ing in sublist))
ingredient_to_idx = {ing: idx for idx, ing in enumerate(all_ingredients)}
idx_to_ingredient = {idx: ing for ing, idx in ingredient_to_idx.items()}

# Функція для перетворення інгредієнтів у multi-hot вектор
def recipe_to_multihot(ingredients):
    vector = np.zeros(len(all_ingredients), dtype=np.float32)
    for ing in ingredients:
        if ing in ingredient_to_idx:
            vector[ingredient_to_idx[ing]] = 1.0
    return vector

# Завантаження моделі
class RecipeModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.encoder(x)
        reconstructed = self.decoder(embedded)
        return reconstructed, embedded

# Ініціалізація моделі
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecipeModel(input_dim=len(all_ingredients))
model.load_state_dict(torch.load("recipe_model.pth", map_location=device))
model.to(device)
model.eval()

# Функція для отримання рекомендацій
def get_recommendations(user_input, top_k=5):
    user_ingredients = [ing.strip().lower() for ing in user_input.split(",")]
    input_vector = recipe_to_multihot(user_ingredients)

    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vector).to(device)
        _, embedding = model(input_tensor.unsqueeze(0))

        similarities = []
        for idx, recipe in enumerate(df["normalized_ingredients"]):
            recipe_vector = recipe_to_multihot(recipe)
            recipe_tensor = torch.FloatTensor(recipe_vector).to(device).unsqueeze(0)
            _, recipe_embedding = model(recipe_tensor)
            sim = cosine_similarity(embedding.cpu().numpy(), recipe_embedding.cpu().numpy())
            similarities.append((idx, sim[0][0]))

    top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return df.iloc[[idx for idx, _ in top_indices]]

# Тестування
user_input = input("Введіть інгредієнти (через кому): ")
recommendations = get_recommendations(user_input)

print("\nРекомендації:")
for _, row in recommendations.iterrows():
    print(f"\nРецепт: {row['title']}")
    print("Інгредієнти:", ", ".join(row['normalized_ingredients']))
