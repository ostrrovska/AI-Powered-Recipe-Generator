import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# =================================================================
# Крок 1: Підготовка даних
# =================================================================

# Завантажити нормалізовані дані
df = pd.read_csv("normalized_recipes.csv")

# Створити словник унікальних інгредієнтів
all_ingredients = list(set(ing for sublist in df["normalized_ingredients"] for ing in sublist))
ingredient_to_idx = {ing: idx for idx, ing in enumerate(all_ingredients)}
idx_to_ingredient = {idx: ing for ing, idx in ingredient_to_idx.items()}


# Функція для перетворення рецептів у multi-hot вектори
def recipe_to_multihot(ingredients):
    vector = np.zeros(len(all_ingredients), dtype=np.float32)
    for ing in ingredients:
        if ing in ingredient_to_idx:
            vector[ingredient_to_idx[ing]] = 1.0
    return vector


# Створити тензори для всіх рецептів
recipe_vectors = np.array([recipe_to_multihot(ing) for ing in df["normalized_ingredients"]])
recipe_tensors = torch.FloatTensor(recipe_vectors)


# =================================================================
# Крок 2: Побудова моделі PyTorch
# =================================================================

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
model = RecipeModel(input_dim=len(all_ingredients))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# =================================================================
# Крок 3: Підготовка DataLoader
# =================================================================

class RecipeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


dataset = RecipeDataset(recipe_tensors)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =================================================================
# Крок 4: Навчання моделі
# =================================================================

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

print("Початок навчання...")
for epoch in range(num_epochs):
    total_loss = 0
    for batch, targets in dataloader:
        batch = batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        reconstructed, _ = model(batch)
        loss = criterion(reconstructed, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Епоха {epoch + 1}/{num_epochs}, Втрати: {total_loss / len(dataloader):.4f}")


# =================================================================
# Крок 5: Функція рекомендацій
# =================================================================

def get_recommendations(user_input, top_k=5):
    # Нормалізація введення
    user_ingredients = [ing.strip().lower() for ing in user_input.split(",")]

    # Створення multi-hot вектора
    input_vector = np.zeros(len(all_ingredients), dtype=np.float32)
    for ing in user_ingredients:
        if ing in ingredient_to_idx:
            input_vector[ingredient_to_idx[ing]] = 1.0

    # Отримання ембеддингів
    with torch.no_grad():
        model.eval()
        input_tensor = torch.FloatTensor(input_vector).to(device)
        _, embedding = model(input_tensor.unsqueeze(0))

        # Обчислення схожості з усіма рецептами
        similarities = []
        for recipe in recipe_tensors:
            _, recipe_embedding = model(recipe.to(device).unsqueeze(0))
            sim = cosine_similarity(embedding.cpu().numpy(), recipe_embedding.cpu().numpy())
            similarities.append(sim[0][0])

    # Топ-K рекомендацій
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices]


# =================================================================
# Крок 6: Тестування системи
# =================================================================

# Приклад використання
user_input = "onion, chicken broth"
recommendations = get_recommendations(user_input)

print("\nРекомендації для:", user_input)
print("=====================================")
for idx, row in recommendations.iterrows():
    print(f"\nРецепт: {row['title']}")
    print("Інгредієнти:", ", ".join(row['normalized_ingredients']))

# Збереження моделі
torch.save(model.state_dict(), "recipe_model.pth")