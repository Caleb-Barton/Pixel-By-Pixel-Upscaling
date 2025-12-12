import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# Configuration
DATA_PATH = "data/kis_model_3x3/"
MODEL_SAVE_PATH = "models/kis_model_3x3/"

# Model hyperparameters
INPUT_SIZE = 9
OUTPUT_SIZE = 9
HIDDEN_LAYERS = [32, 81, 32]
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100

# Training settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UpscaleNet(nn.Module):
    """
    Fully connected neural network for pixel upscaling.
    """
    def __init__(self, input_size, hidden_layers, output_size):
        super(UpscaleNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_data():
    """
    Load training and test data from .npy files.
    Returns DataLoaders for training and testing.
    """
    print("Loading data...")
    
    # Load numpy arrays
    train_inputs = np.load(os.path.join(DATA_PATH, "train_inputs.npy"))
    train_outputs = np.load(os.path.join(DATA_PATH, "train_outputs.npy"))
    test_inputs = np.load(os.path.join(DATA_PATH, "test_inputs.npy"))
    test_outputs = np.load(os.path.join(DATA_PATH, "test_outputs.npy"))
    
    # Normalize to [0, 1] range
    train_inputs = train_inputs / 255.0
    train_outputs = train_outputs / 255.0
    test_inputs = test_inputs / 255.0
    test_outputs = test_outputs / 255.0
    
    # Convert to PyTorch tensors
    train_inputs = torch.FloatTensor(train_inputs)
    train_outputs = torch.FloatTensor(train_outputs)
    test_inputs = torch.FloatTensor(test_inputs)
    test_outputs = torch.FloatTensor(test_outputs)
    
    # Create datasets
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on a dataset.
    Returns the average loss.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def train_model():
    """
    Train the upscaling neural network.
    """
    # Create model save directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize model
    model = UpscaleNet(INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE).to(DEVICE)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTraining on: {DEVICE}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_test_loss = float('inf')
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = evaluate_model(model, test_loader, criterion)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
        
        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f} | "
              f"Best Test: {best_test_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "last_model.pth"))
    
    print("="*60)
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Models saved to: {MODEL_SAVE_PATH}")
    
    # Convert losses back to pixel space (0-255)
    print(f"\nIn pixel space (0-255):")
    print(f"Best test MSE: {best_test_loss * (255**2):.4f}")
    print(f"Best test RMSE: {np.sqrt(best_test_loss) * 255:.4f}")


if __name__ == "__main__":
    train_model()