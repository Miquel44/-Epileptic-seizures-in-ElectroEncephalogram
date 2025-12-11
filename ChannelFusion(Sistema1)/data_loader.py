import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_PATH, BATCH_SIZE, TRAIN_SPLIT


class EEGDataset(Dataset):
    """Dataset para señales EEG con fusión de canales."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

def load_single_patient(patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga datos de un paciente específico."""
    eeg_file = DATA_PATH / f"{patient_id}_seizure_EEGwindow_1.npz"
    metadata_file = DATA_PATH / f"{patient_id}_seizure_metadata_1.parquet"

    # Cargar señales EEG
    eeg_data = np.load(eeg_file, allow_pickle=True)
    signals = eeg_data['EEG_win']

    # Cargar etiquetas (ya son 0/1)
    metadata = pd.read_parquet(metadata_file)
    labels = metadata['class'].values

    return signals, labels




def load_all_patients() -> Tuple[np.ndarray, np.ndarray]:
    """Carga datos de todos los pacientes disponibles."""
    all_signals = []
    all_labels = []

    # Buscar todos los archivos de pacientes
    patient_files = list(DATA_PATH.glob("chb*_seizure_EEGwindow_1.npz"))

    for eeg_file in patient_files:
        patient_id = eeg_file.stem.replace("_seizure_EEGwindow_1", "")
        try:
            signals, labels = load_single_patient(patient_id)
            all_signals.append(signals)
            all_labels.append(labels)
            print(f"Cargado {patient_id}: {signals.shape}")
        except Exception as e:
            print(f"Error cargando {patient_id}: {e}")

    return np.concatenate(all_signals), np.concatenate(all_labels)


def create_channel_fusion_input(signals: np.ndarray) -> np.ndarray:
    """
    Prepara datos para fusión de canales.
    Entrada: (samples, channels, time_points)
    Salida: (samples, 1, channels, time_points) - formato imagen
    """
    if signals.ndim == 2:
        # Si es (samples, time_points), expandir
        signals = signals[:, np.newaxis, :]

    # Añadir dimensión de canal para CNN (como imagen 2D)
    signals = signals[:, np.newaxis, :, :]
    return signals


def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """Crea DataLoaders para entrenamiento y validación."""
    signals, labels = load_all_patients()
    print(f"Datos originales cargados: {signals.shape}, Etiquetas: {labels.shape}")
    signals = create_channel_fusion_input(signals)
    print(f"Datos totales cargados: {signals.shape}, Etiquetas: {labels.shape}")
    X_train, X_val, y_train, y_val = train_test_split(
        signals, labels, test_size=1 - TRAIN_SPLIT, random_state=42, stratify=labels
    )
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader
