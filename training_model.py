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

def preprocess_data():
    # Load and preprocess data
    df = pd.read_csv('db/recipes.csv')
    
    # Convert ingredients to a format suitable for CountVectorizer
    ingredients_text = df['RecipeIngredientParts'].fillna('').str.lower()
    
    # Create vocabulary with minimum frequency
    vectorizer = CountVectorizer(
        min_df=Config.MIN_INGREDIENT_FREQ,
        token_pattern=r'[a-zA-Z]+',
        binary=True
    )
    
    # Fit and transform in batches to save memory
    ingredient_matrix = vectorizer.fit_transform(ingredients_text)
    
    # Save vocabulary for later use
    vocab = vectorizer.vocabulary_
    with open('ingredient_vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    return ingredient_matrix, vocab

class RecipeDataset(Dataset):
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix
    
    def __len__(self):
        return self.sparse_matrix.shape[0]
    
    def __getitem__(self, idx):
        # Convert sparse row to dense tensor efficiently
        x = torch.FloatTensor(self.sparse_matrix[idx].toarray()[0])
        return x, x

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

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(Config.DEVICE)
            target = target.to(Config.DEVICE)
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch, (data, target) in enumerate(progress_bar):
        data = data.to(Config.DEVICE)
        target = target.to(Config.DEVICE)
        
        optimizer.zero_grad()
        reconstructed, _ = model(data)
        loss = criterion(reconstructed, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({'train_loss': total_loss / (batch + 1)})
    
    # Calculate validation loss
    val_loss = evaluate_model(model, val_loader, criterion)
    return total_loss / len(train_loader), val_loss

def main():
    print("Preprocessing data...")
    ingredient_matrix, vocab = preprocess_data()
    
    # Create full dataset
    dataset = RecipeDataset(ingredient_matrix)
    
    # Split into train and validation
    val_size = int(len(dataset) * Config.VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # Initialize model
    model = RecipeAutoencoder(len(vocab)).to(Config.DEVICE)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=0.01  # L2 regularization
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * Config.NUM_EPOCHS
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training... Using device: {Config.DEVICE}")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler)
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'best_recipe_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
    gc.collect()  # Clean up memory