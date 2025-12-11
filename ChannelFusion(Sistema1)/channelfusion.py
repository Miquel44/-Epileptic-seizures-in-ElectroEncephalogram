"""
Sistema 1: Channel Fusion para detección de convulsiones epilépticas.
Basado en el enfoque de fusión de canales EEG usando CNN.
"""

from config import *
from data_loader import get_data_loaders, load_single_patient
from models import ChannelFusionCNN, ChannelFusionLSTM
from trainer import Trainer

print(f"Dispositivo: {DEVICE}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
def main():
    print("=== Sistema 1: Channel Fusion ===\n")

    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader = get_data_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # Crear modelo
    model = ChannelFusionCNN()
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")

    # Entrenar
    trainer = Trainer(model)
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)

    # Evaluar
    results = trainer.evaluate(val_loader)
    print(f"\nAccuracy final: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")

    # Guardar modelo
    trainer.save_model("channel_fusion_model.pth")
    print("\nModelo guardado en 'channel_fusion_model.pth'")


if __name__ == "__main__":
    main()
