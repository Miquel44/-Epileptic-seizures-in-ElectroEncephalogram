import torch
from pathlib import Path

# Dispositivo (GPU si está disponible)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rutas
DATA_PATH = Path(r"C:\Users\mique\OneDrive\Escritorio\Proyectos\-Epileptic-seizures-in-ElectroEncephalogram\Data\input\input")

# Parámetros del modelo
NUM_CHANNELS = 21
WINDOW_SIZE = 128
NUM_CLASSES = 2

# Entrenamiento
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
TRAIN_SPLIT = 0.8
