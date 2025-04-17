import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .base import BaseModel

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# TransformerBlock: Implementation of a single transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=n_features, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ff = nn.Sequential(
            nn.Linear(n_features, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, n_features)
        )
        self.norm2 = nn.LayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Self-attention and residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward network and residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
    
# TransformerClassifier: Simple Transformer-based classifier for sequence inputs
class TransformerClassifier(nn.Module):
    def __init__(self, sequence_length, n_features, num_heads, ff_dim, dropout_rate):
        super(TransformerClassifier, self).__init__()
        self.transformer = TransformerBlock(n_features, num_heads, ff_dim, dropout_rate)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_features, 1)  # Binary classification


    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)
        x = self.transformer(x) # Shape: (B, T, F)
        x = x.transpose(1,2) # Shape: (B, F, T)
        x = self.pool(x).squeeze(-1)  # Shape: (B, F)
        x = self.fc(x) # Shape: (B, 1)
        return torch.sigmoid(x)
        
class TransformerModel(BaseModel):
    def __init__(self, sequence_length=10, n_features=16, num_heads=4, ff_dim=128, dropout_rate=0.1, batch_size=32, epochs=10, lr=1e-3, device=None):
        """
        Initialize the Transformer model with optional parameters.

        Parameters:
        sequence_length (int): Length of the input sequences.
        n_features (int): Number of features in the input data.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward network.
        dropout_rate (float): Dropout rate for regularization.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerClassifier(sequence_length, n_features, num_heads, ff_dim, dropout_rate).to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Early stopping configs
        self.early_stopping = True
        self.patience = 100
        self.no_improve_epochs = 0
        self.checkpoint_path = "transformer_model_checkpoint.pth"
        self.best_loss = float('inf')
    
    def _preprocess_x(self, X: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess the input DataFrame into a PyTorch tensor.

        Parameters:
        X (pd.DataFrame): Input feature set.

        Returns:
        torch.Tensor: Preprocessed tensor.
        """
        
        X_np = X.values
        X_np = self.scaler.fit_transform(X_np)
        B = X_np.shape[0]
        T = self.sequence_length
        F = self.n_features

        # Check if the number of features matches the expected shape
        if X_np.shape[1] != self.sequence_length * self.n_features:
            raise ValueError(f"Expected {self.sequence_length * self.n_features} features, but got {X_np.shape[1]}")

        X_reshaped = X_np.reshape(B, T, F)
        return torch.tensor(X_reshaped, dtype=torch.float32).to(self.device)
    
    def _preprocess_y(self, y: pd.Series) -> torch.Tensor:
        """
        Preprocess the target variable into a PyTorch tensor.

        Parameters:
        y (pd.Series): Target variable.

        Returns:
        torch.Tensor: Preprocessed tensor.
        """
        return torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Parameters:
        X (pd.DataFrame): Feature set for training.
        y (pd.Series): Target variable for training.
        """

        X_tensor = self._preprocess_x(X)
        y_tensor = self._preprocess_y(y)


        X_np = X_tensor.cpu().numpy()
        y_np = y_tensor.cpu().numpy()

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
        )

        # Convert back to tensors
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(self.device)
        X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).to(self.device)

        # Create loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.model.train()
        for epoch in range(self.epochs):

            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y.float())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()  # Add batch loss to epoch loss
            avg_train_loss = epoch_loss / len(train_loader)

            # Validatiion
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping logic
            if self.early_stopping:
                if avg_val_loss < self.best_loss:
                    self.best_loss = avg_val_loss
                    self.no_improve_epochs = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    logger.info("Best model saved.")
                else:
                    self.no_improve_epochs += 1
                    logger.info(f"No improvement. Patience: {self.no_improve_epochs}/{self.patience}")
                    if self.no_improve_epochs >= self.patience:
                        logger.info("Early stopping triggered.")
                        break
            else:
                torch.save(self.model.state_dict(), f"transformer_epoch_{epoch+1}.pt")
    
    def predict(self, X:pd.DataFrame) -> pd.Series: 
        """
        Parameters:
        X (pd.DataFrame): Feature set for prediction.

        Returns:
        pd.Series: Predicted values.
        """
        if self.early_stopping and os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
        X_tensor = self._preprocess_x(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).view(-1)
            predictions = (outputs > 0.5).float().cpu().numpy()
        return pd.Series(predictions)