import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gc
import random

from config import DATA_PATH, BATCH_SIZE, TRAIN_SPLIT


## Datasets

class EEGDataset(Dataset):
    """Dataset optimizado para PyTorch (float32)."""
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data) # from_numpy evita copia extra si ya es array
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

class EEGSequenceDataset(Dataset):
    """
    Devuelve secuencias de K ventanas consecutivas:
      X_seq: [K, 1, C, T]
      y_out: etiqueta (int64)

    label_mode:
      - "last": etiqueta de la última ventana
      - "any":  1 si alguna de las K ventanas es crisis
    stride:
      - paso entre secuencias (1 = secuencias solapadas)
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray, seq_len: int = 5,
                 label_mode: str = "last", stride: int = 1):
        assert data.ndim == 4, f"data debe ser (N,1,C,T). Got {data.shape}"
        assert len(data) == len(labels), "data y labels deben tener misma longitud"
        assert seq_len >= 1, "seq_len debe ser >= 1"
        assert stride >= 1, "stride debe ser >= 1"
        assert label_mode in ("last", "any"), "label_mode debe ser 'last' o 'any'"

        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.seq_len = seq_len
        self.label_mode = label_mode
        self.stride = stride

        # Nº de secuencias posibles con stride
        self.n_seq = (len(self.labels) - seq_len) // stride + 1 if len(self.labels) >= seq_len else 0

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_len

        x_seq = self.data[start:end]       # [K,1,C,T]
        y_seq = self.labels[start:end]     # [K]

        if self.label_mode == "last":
            y_out = y_seq[-1]
        else:  # "any"
            y_out = (y_seq.max() > 0).long()

        return x_seq, y_out

## Funcions

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

def get_patient_dataloader(patient_id: str, batch_size=BATCH_SIZE,
                           sequential: bool = False, seq_len: int = 5,
                           label_mode: str = "last", stride: int = 1):
    """
    MODO PERSONALIZADO
    - Normal: devuelve ventanas individuales (EEGDataset)
    - Secuencial: devuelve secuencias de ventanas (EEGSequenceDataset)
    """
    print(f"--- [Personalized] Cargando: {patient_id} ---", flush=True)

    signals, labels = load_single_patient(patient_id)
    signals = create_channel_fusion_input(signals)

    split_idx = int(len(signals) * TRAIN_SPLIT)

    X_train, y_train = signals[:split_idx], labels[:split_idx]
    X_val, y_val = signals[split_idx:], labels[split_idx:]

    # Balanceo SOLO si NO es secuencial o si aceptas perder orden temporal
    # (tu balanceo actual rompe continuidad temporal -> secuencias menos "reales")
    if not sequential:
        idx_normal = np.where(y_train == 0)[0]
        idx_seizure = np.where(y_train == 1)[0]
        n_seizure = len(idx_seizure)
        n_normal = len(idx_normal)

        if n_seizure > 0 and n_normal > n_seizure:
            np.random.seed(42)
            idx_normal_sampled = np.random.choice(idx_normal, size=n_seizure, replace=False)
            idx_balanced = np.sort(np.concatenate([idx_normal_sampled, idx_seizure]))
            X_train = X_train[idx_balanced]
            y_train = y_train[idx_balanced]
            print(f"   [BALANCED] Train reducido: {n_normal} -> {n_seizure} normales", flush=True)
    else:
        # Nota importante: para LSTM secuencial lo ideal es NO submuestrear así,
        # porque reduces continuidad temporal. Para probar que ejecuta, lo dejamos sin balanceo.
        pass

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | "
          f"Crisis Train: {y_train.sum()} ({100*y_train.sum()/len(y_train):.1f}%) | "
          f"Crisis Val: {y_val.sum()}", flush=True)

    if sequential:
        train_ds = EEGSequenceDataset(X_train, y_train, seq_len=seq_len, label_mode=label_mode, stride=stride)
        val_ds   = EEGSequenceDataset(X_val, y_val,   seq_len=seq_len, label_mode=label_mode, stride=stride)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"   [WARN] No hay suficientes ventanas para construir secuencias con seq_len={seq_len}.", flush=True)
            # fallback: dataset normal para que no pete
            train_ds = EEGDataset(X_train, y_train)
            val_ds   = EEGDataset(X_val, y_val)
    else:
        train_ds = EEGDataset(X_train, y_train)
        val_ds   = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

# def get_patient_dataloader(patient_id: str, batch_size=BATCH_SIZE):
#     """
#     MODO PERSONALIZADO
#     Entrena con el pasado de UN paciente, valida con su futuro.
#     """
#     print(f"--- [Personalized] Cargando: {patient_id} ---", flush=True)
    
#     signals, labels = load_single_patient(patient_id)
#     signals = create_channel_fusion_input(signals)
    
#     # Split cronológico (80/20) sin mezclar para evitar leakage temporal
#     split_idx = int(len(signals) * TRAIN_SPLIT)
    
#     X_train, y_train = signals[:split_idx], labels[:split_idx]
#     X_val, y_val = signals[split_idx:], labels[split_idx:]

#     # Índices de cada clase en train
#     idx_normal = np.where(y_train == 0)[0]
#     idx_seizure = np.where(y_train == 1)[0]
    
#     n_seizure = len(idx_seizure)
#     n_normal = len(idx_normal)
    
#     if n_seizure > 0 and n_normal > n_seizure:
#         # Submuestrear normales para igualar a seizures
#         np.random.seed(42)  # Reproducibilidad
#         idx_normal_sampled = np.random.choice(idx_normal, size=n_seizure, replace=False)
        
#         # Combinar y ordenar índices (mantener orden temporal)
#         idx_balanced = np.sort(np.concatenate([idx_normal_sampled, idx_seizure]))
        
#         X_train = X_train[idx_balanced]
#         y_train = y_train[idx_balanced]
        
#         print(f"   [BALANCED] Train reducido: {n_normal} -> {n_seizure} normales", flush=True)
    
#     # print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Crisis Train: {y_train.sum()} | Crisis Val: {y_val.sum()}", flush=True)
#     print(f"   Train: {len(X_train)} | Val: {len(X_val)} | "
#           f"Crisis Train: {y_train.sum()} ({100*y_train.sum()/len(y_train):.1f}%) | "
#           f"Crisis Val: {y_val.sum()}", flush=True)

#     # Entrenando PODEMOS hacer shuffle del pasado
#     train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
#     return train_loader, val_loader

# def get_patient_dataloader(patient_id: str, batch_size=BATCH_SIZE):
#     """
#     MODO PERSONALIZADO
#     Split estratificado para mantener proporción de crisis.
#     """
#     print(f"--- [Personalized] Cargando: {patient_id} ---", flush=True)
    
#     signals, labels = load_single_patient(patient_id)
#     signals = create_channel_fusion_input(signals)
    
#     # Split ESTRATIFICADO (mantiene proporción de clases en ambos conjuntos)
#     X_train, X_val, y_train, y_val = train_test_split(
#         signals, labels,
#         test_size=(1 - TRAIN_SPLIT),  # 20% val
#         stratify=labels,               # Mantener proporción de crisis
#         random_state=42
#     )
    
#     print(f"   Train: {len(X_train)} | Val: {len(X_val)} | "
#           f"Crisis Train: {y_train.sum()} ({100*y_train.sum()/len(y_train):.1f}%) | "
#           f"Crisis Val: {y_val.sum()} ({100*y_val.sum()/len(y_val):.1f}%)", flush=True)

#     train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=batch_size, 
#                               shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=batch_size, 
#                             shuffle=False, num_workers=4, pin_memory=True)
    
#     return train_loader, val_loader


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

    idx_normal = np.where(train_y == 0)[0]
    idx_seizure = np.where(train_y == 1)[0]
    
    n_seizure = len(idx_seizure)
    n_normal = len(idx_normal)
    
    if n_seizure > 0 and n_normal > n_seizure:
        # Submuestrear normales para igualar a seizures
        np.random.seed(42)  # Reproducibilidad
        idx_normal_sampled = np.random.choice(idx_normal, size=n_seizure, replace=False)
        
        # Combinar índices (NO ordenamos para mantener mezcla de pacientes)
        idx_balanced = np.concatenate([idx_normal_sampled, idx_seizure])
        np.random.shuffle(idx_balanced)  # Mezclar para que no estén agrupados
        
        train_X = train_X[idx_balanced]
        train_y = train_y[idx_balanced]
        
        print(f"   [BALANCED] Train: {n_normal + n_seizure} -> {len(train_X)} "
                f"(Normal: {n_normal} -> {n_seizure}, Seizure: {n_seizure})", flush=True)
    
    # print(f"   Train: {len(train_X)} | Test: {len(test_X)}", flush=True)
    print(f"   Train: {len(train_X)} ({train_y.sum()} seizure, {100*train_y.sum()/len(train_y):.1f}%) | "
          f"Test: {len(test_X)} ({test_y.sum()} seizure)", flush=True)

    train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(EEGDataset(test_X, test_y), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

# Función deprecated para seguridad
def get_data_loaders(*args, **kwargs):
    raise DeprecationWarning("Usar get_patient_dataloader o get_leave_one_out_dataloader")