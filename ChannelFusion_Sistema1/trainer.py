import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict
import numpy as np

from config import LEARNING_RATE, DEVICE

class Trainer:
    def __init__(self, model, patience=5):
        self.model = model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.patience = patience
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.best_model_state = None

    def train(self, train_loader, val_loader, epochs=50):
        print(f"Entrenando en {DEVICE}...", flush=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y.size(0)
                train_correct += predicted.eq(y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # --- Validation ---
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += y.size(0)
                    val_correct += predicted.eq(y).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Ep {epoch + 1}/{epochs} | "
                  f"TrLoss: {train_loss:.4f} TrAcc: {train_acc:.4f} | "
                  f"ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}", flush=True)

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping! (Patience {self.patience})", flush=True)
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        # Cargar mejor modelo al acabar si no paró por early stopping justo al final
        if self.best_model_state:
             self.model.load_state_dict(self.best_model_state)
             
        return self.history

    def evaluate(self, test_loader) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(DEVICE)
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Usar zero_division=0 para evitar warnings si no predice ninguna crisis
        report = classification_report(all_labels, all_preds, 
                                     target_names=['Normal', 'Seizure'], 
                                     zero_division=0)
        print("\n=== Reporte Validación ===", flush=True)
        print(report, flush=True)

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)