import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, Tuple
import numpy as np

from config import EPOCHS, LEARNING_RATE


class Trainer:
    """Clase para entrenar y evaluar modelos de fusión de canales."""

    def __init__(self, model: nn.Module, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0

        for data, labels in train_loader:
            data, labels = data.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evalúa el modelo en validación."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = EPOCHS) -> Dict:
        """Entrena el modelo completo."""
        print(f"Entrenando en {self.device}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Época {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc:.4f}")

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluación final con métricas detalladas."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print("\n=== Resultados ===")
        print(classification_report(all_labels, all_preds,
                                    target_names=['No Seizure', 'Seizure']))

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
