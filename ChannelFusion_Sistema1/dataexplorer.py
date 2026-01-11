import numpy as np
import pandas as pd
from pathlib import Path



DATA_PATH = Path(r"C:\Users\mique\OneDrive\Escritorio\Proyectos\-Epileptic-seizures-in-ElectroEncephalogram\Data\input\input")

metadata = pd.read_parquet(DATA_PATH / "chb01_seizure_metadata_1.parquet")
print("Valores únicos de 'class':", metadata['class'].unique())
print("\nDistribución:")
print(metadata['class'].value_counts())


DATA_PATH = Path(r"C:\Users\mique\OneDrive\Escritorio\Proyectos\-Epileptic-seizures-in-ElectroEncephalogram\Data\input\input")

# Cargar datos de un paciente
eeg = np.load(DATA_PATH / "chb01_seizure_EEGwindow_1.npz", allow_pickle=True)
metadata = pd.read_parquet(DATA_PATH / "chb01_seizure_metadata_1.parquet")

print("Shape señales:", eeg['EEG_win'].shape)
print("Número de ventanas:", len(metadata))
print("\nDistribución de clases por ventana:")
print(metadata['class'].value_counts())
print("\nEjemplo de metadata:")
print(metadata.head(10))
