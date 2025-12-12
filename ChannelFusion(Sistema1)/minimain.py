"""
Sistema 1: Channel Fusion para detección de convulsiones epilépticas.
Versión reducida: solo 4 pacientes aleatorios.
"""

import random
import matplotlib.pyplot as plt
from config import *
from data_loader import get_data_loaders
from models import ChannelFusionCNN
from trainer import Trainer

print(f"Dispositivo: {DEVICE}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def plot_training_history(history, save_path="training_plots.png"):
    """Genera gráficas del entrenamiento."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Pérdida durante entrenamiento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy durante entrenamiento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Train vs Val Loss comparison
    axes[1, 0].bar(['Train', 'Val'], [history['train_loss'][-1], history['val_loss'][-1]],
                   color=['blue', 'red'])
    axes[1, 0].set_title('Loss final')
    axes[1, 0].set_ylabel('Loss')

    # 4. Train vs Val Accuracy comparison
    axes[1, 1].bar(['Train', 'Val'], [history['train_acc'][-1], history['val_acc'][-1]],
                   color=['blue', 'red'])
    axes[1, 1].set_title('Accuracy final')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráficas guardadas en '{save_path}'")


def main():
    print("=== Sistema 1: Channel Fusion (4 pacientes) ===\n")

    # Seleccionar 4 pacientes aleatorios
    all_patients = [f"chb{i:02d}" for i in range(1, 24)]
    selected_patients = random.sample(all_patients, 4)
    print(f"Pacientes seleccionados: {selected_patients}\n")

    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader = get_data_loaders(patients=selected_patients)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Crear modelo
    model = ChannelFusionCNN()
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")

    # Entrenar con early stopping (patience=5)
    trainer = Trainer(model, patience=5)
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)

    # Evaluar
    results = trainer.evaluate(val_loader)
    print(f"\nAccuracy final: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")

    # Plotear gráficas
    plot_training_history(history)

    # Guardar modelo
    trainer.save_model("channel_fusion_model_4patients.pth")
    print("\nModelo guardado en 'channel_fusion_model_4patients.pth'")


if __name__ == "__main__":
    main()
