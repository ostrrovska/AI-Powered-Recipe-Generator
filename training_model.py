import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import gc
from tqdm import tqdm
from parse_parquet import load_parquet_file
import re

# Configuration
class Config:
    BATCH_SIZE = 64  # Increased for better GPU utilization
    LEARNING_RATE = 3e-4  # Standard learning rate for Adam
    NUM_EPOCHS = 10
    EMBEDDING_DIM = 256  # Increased for better representation
    HIDDEN_DIM = 512
    MIN_INGREDIENT_FREQ = 5  # Minimum frequency for ingredients
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # For DataLoader
    VALIDATION_SPLIT = 0.1  # 10% for validation
    EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs

class RecipeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Autoencoder: input = target

class RecipeAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),  # Better than BatchNorm for this case
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM, Config.EMBEDDING_DIM)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(Config.EMBEDDING_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.HIDDEN_DIM, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embedded = self.encoder(x)
        reconstructed = self.decoder(embedded)
        return reconstructed, embedded

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler):
    model.train()
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(Config.DEVICE)
        batch_y = batch_y.to(Config.DEVICE)
        
        optimizer.zero_grad()
        output, _ = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    return total_train_loss / len(train_loader)

def validate_model(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(Config.DEVICE)
            batch_y = batch_y.to(Config.DEVICE)
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_loader)

def preprocess_data():
    # Load and preprocess data from Parquet
    print("Loading data from Parquet...")
    df = load_parquet_file()
    if df is None:
        raise RuntimeError("Failed to load recipe data")
    
    # Process ingredients - they are already numpy arrays
    print("Processing ingredients...")
    all_ingredients = df['RecipeIngredientParts'].apply(lambda x: ', '.join(x.astype(str))).str.lower()
    
    # Create vocabulary from ingredients
    print("Creating vocabulary...")
    vectorizer = CountVectorizer(
        min_df=5,  # Require ingredient to appear in at least 5 recipes
        max_df=0.8,  # Ignore ingredients that appear in >80% of recipes
        stop_words=None,  # Don't remove any stop words
        max_features=5000  # Limit vocabulary size
    )
    vectorizer.fit(all_ingredients)
    
    # Save vocabulary for later use
    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}  # Convert numpy.int64 to int
    with open('ingredient_vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    # Transform ingredients to sparse matrix
    print("Vectorizing ingredients...")
    ingredient_matrix = vectorizer.transform(all_ingredients)
    
    # Convert to dense tensor and normalize to [0,1]
    ingredient_tensor = torch.FloatTensor(ingredient_matrix.toarray())
    ingredient_tensor = torch.clamp(ingredient_tensor / ingredient_tensor.max(), 0, 1)  # Normalize to [0,1]
    
    # Create dataset
    dataset = RecipeDataset(ingredient_tensor)
    val_size = int(len(dataset) * Config.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    return train_loader, val_loader, len(vocab)

def main():
    # Preprocess data
    train_loader, val_loader, input_dim = preprocess_data()
    
    # Initialize model
    model = RecipeAutoencoder(input_dim).to(Config.DEVICE)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler)
        val_loss = validate_model(model, val_loader, criterion)
        
        # Step the scheduler once per epoch
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_recipe_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
    
    print("Training completed!")

if __name__ == "__main__":
    main()
    gc.collect()  # Clean up memory