import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import gc
import random

from config import DATA_PATH, BATCH_SIZE, TRAIN_SPLIT


class EEGDataset(Dataset):
    """Dataset optimizado para PyTorch (float32)."""
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data) # from_numpy evita copia extra si ya es array
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def load_single_patient(patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga un paciente gestionando memoria eficientemente."""
    eeg_file = DATA_PATH / f"{patient_id}_seizure_EEGwindow_1.npz"
    metadata_file = DATA_PATH / f"{patient_id}_seizure_metadata_1.parquet"

    if not eeg_file.exists():
        raise FileNotFoundError(f"No existe: {eeg_file}")

    # Cargar EEG en float32 directamente usando 'with' para cerrar fichero rápido
    with np.load(eeg_file, allow_pickle=True) as eeg_data:
        signals = eeg_data['EEG_win'].astype(np.float32)

    # Cargar Metadata
    # engine='auto' suele ir bien si pyarrow está instalado
    metadata = pd.read_parquet(metadata_file, engine='fastparquet')
    labels = metadata['class'].values.astype(np.int64)

    return signals, labels


def create_channel_fusion_input(signals: np.ndarray) -> np.ndarray:
    """Prepara (N, Channels, Time) -> (N, 1, Channels, Time) para CNN 2D."""
    # Asegurar que entra como (N, Ch, T)
    if signals.ndim == 2:
        signals = signals[:, np.newaxis, :] # (N, 1, T) si falta canal
    
    # Añadir dimensión de canal de 'imagen' para Conv2d: (N, 1, Ch, T)
    if signals.ndim == 3:
        signals = signals[:, np.newaxis, :, :]
        
    return signals


def load_data_from_patient_list(patient_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Carga y concatena datos de varios pacientes (Para modo Leave-One-Out)."""
    signals_list = []
    labels_list = []
    n_channels = 21

    print(f"   -> Cargando bloque de {len(patient_list)} pacientes...", flush=True)

    for i, patient_id in enumerate(patient_list, 1):
        try:
            sig, lab = load_single_patient(patient_id)
            
            # Estandarizar 21 canales
            if sig.shape[1] > n_channels:
                sig = sig[:, :n_channels, :]
            elif sig.shape[1] < n_channels:
                padding = np.zeros((sig.shape[0], n_channels - sig.shape[1], sig.shape[2]), dtype=np.float32)
                sig = np.concatenate([sig, padding], axis=1)

            signals_list.append(sig)
            labels_list.append(lab)
            
            # Liberar inmediatamante
            del sig, lab
            
        except Exception as e:
            print(f"      [WARN] Skip {patient_id}: {e}", flush=True)

    gc.collect()

    if not signals_list:
        raise ValueError("No se pudieron cargar datos.")

    # Concatenar todo
    X = np.concatenate(signals_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    del signals_list, labels_list
    gc.collect()
    
    # Preparar forma
    X = create_channel_fusion_input(X)
    
    return X, y

# la funcion de arriba pero mejor??:
def preload_all_patients(patient_list: List[str]) -> dict:
    """
    Carga todos los pacientes en un diccionario {patient_id: (X, y)}.
    X ya tiene forma (N, 1, 21, 128) lista para CNN.
    """
    patient_data = {}
    n_channels = 21

    print(f"   Precargando {len(patient_list)} pacientes...", flush=True)

    for patient_id in patient_list:
        try:
            sig, lab = load_single_patient(patient_id)
            
            if sig.shape[1] > n_channels:
                sig = sig[:, :n_channels, :]
            elif sig.shape[1] < n_channels:
                padding = np.zeros((sig.shape[0], n_channels - sig.shape[1], sig.shape[2]), dtype=np.float32)
                sig = np.concatenate([sig, padding], axis=1)
            
            patient_data[patient_id] = (
                create_channel_fusion_input(sig.astype(np.float32)),
                lab.astype(np.int64)
            )
            print(f"      {patient_id}: {sig.shape[0]} ventanas", flush=True)
            
        except Exception as e:
            print(f"      [WARN] Skip {patient_id}: {e}", flush=True)

    print(f"   [OK] {len(patient_data)} pacientes listos.\n", flush=True)
    return patient_data


# --- MODOS DE CARGA ---

def get_patient_dataloader(patient_id: str, batch_size=BATCH_SIZE):
    """
    MODO PERSONALIZADO
    Entrena con el pasado de UN paciente, valida con su futuro.
    """
    print(f"--- [Personalized] Cargando: {patient_id} ---", flush=True)
    
    signals, labels = load_single_patient(patient_id)
    signals = create_channel_fusion_input(signals)
    
    # Split cronológico (80/20) sin mezclar para evitar leakage temporal
    split_idx = int(len(signals) * TRAIN_SPLIT)
    
    X_train, y_train = signals[:split_idx], labels[:split_idx]
    X_val, y_val = signals[split_idx:], labels[split_idx:]
    
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Crisis Train: {y_train.sum()} | Crisis Val: {y_val.sum()}", flush=True)

    # Entrenando PODEMOS hacer shuffle del pasado
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader


def get_leave_one_out_dataloader(train_patients: List[str], test_patient: str, 
                                  batch_size=BATCH_SIZE, 
                                  preloaded_data: dict = None):
    """
    MODO GENERAL (LEAVE-ONE-OUT)
    Entrena con TODOS menos uno. Valida con el paciente NUEVO.
    
    Si preloaded_data es un dict {patient_id: (X, y)}, lo usa en vez de cargar de disco.
    """
    print(f"--- [Leave-One-Out] Test Patient: {test_patient} ---")
    
    if preloaded_data is not None:
        # MODO RÁPIDO: Usar datos precargados
        print("   Usando datos precargados...", flush=True)
        
        train_X = np.concatenate([X for pid, (X, y) in preloaded_data.items() if pid != test_patient])
        train_y = np.concatenate([y for pid, (X, y) in preloaded_data.items() if pid != test_patient])
        
        test_X, test_y = preloaded_data[test_patient]
        
    else:
        # MODO ORIGINAL: Cargar de disco
        print("   Cargando Train Set (Multipaciente)...", flush=True)
        train_X, train_y = load_data_from_patient_list(train_patients)
        
        print(f"   Cargando Test Patient...", flush=True)
        test_X, test_y = load_data_from_patient_list([test_patient])
    
    print(f"   Train: {len(train_X)} | Test: {len(test_X)}", flush=True)

    train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(EEGDataset(test_X, test_y), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader

# Función deprecated para seguridad
def get_data_loaders(*args, **kwargs):
    raise DeprecationWarning("Usar get_patient_dataloader o get_leave_one_out_dataloader")