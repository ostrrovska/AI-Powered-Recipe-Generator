import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Завантажити нормалізовані дані
df = pd.read_csv("normalized_recipes.csv")
df["normalized_ingredients"] = df["normalized_ingredients"].apply(lambda x: x.split())


# Створити словник унікальних інгредієнтів
all_ingredients = list(set(ing for sublist in df["normalized_ingredients"] for ing in sublist))
ingredient_to_idx = {ing: idx for idx, ing in enumerate(all_ingredients)}

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

# Клас моделі
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

# DataLoader
class RecipeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

dataset = RecipeDataset(recipe_tensors)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Навчання
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 40

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

# Збереження моделі
torch.save(model.state_dict(), "recipe_model.pth")
print("Модель збережено у recipe_model.pth")