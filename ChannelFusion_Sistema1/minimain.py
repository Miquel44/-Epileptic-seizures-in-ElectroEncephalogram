"""
Sistema 1: Channel Fusion
-------------------------
Ejecutar dos veces cambiando MODE:
1. MODE = 'personalized' -> Baseline específico de paciente.
2. MODE = 'leave_one_out' -> Baseline generalista (comparativa).
"""
import sys
import pandas as pd
import numpy as np
import torch
import os
import random

from config import *
from data_loader import get_patient_dataloader, get_leave_one_out_dataloader, preload_all_patients
from models import ChannelFusionCNN
from trainer import Trainer

# >>> CONFIGURACIÓN DEL EXPERIMENTO <<<
# MODE = 'personalized' 
MODE = 'leave_one_out'

def run_personalized_experiment(patients):
    print(f"\n>>> INICIANDO EXPERIMENTO PERSONALIZADO (Total: {len(patients)}) <<<")
    results_table = []

    for patient_id in patients:
        print(f"\n=== Paciente {patient_id} ===", flush=True)
        try:
            # 1. Splits temporales
            train_loader, val_loader = get_patient_dataloader(patient_id)
            
            # Verificar datos suficientes
            if len(train_loader) == 0: 
                print("SKIP: Datos vacíos.")
                continue

            # 2. Entrenar
            model = ChannelFusionCNN()
            trainer = Trainer(model, patience=5)
            trainer.train(train_loader, val_loader, epochs=EPOCHS)
            
            # 3. Evaluar
            metrics = trainer.evaluate(val_loader)
            metrics['patient'] = patient_id
            results_table.append(metrics)
            
            # 4. Guardar
            trainer.save_model(f"models/personalized_{patient_id}.pth")
            
            # Limpieza
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR FATAL en {patient_id}: {e}", flush=True)

    return results_table


def run_leave_one_out_experiment(patients):
    print(f"\n>>> INICIANDO EXPERIMENTO LEAVE-ONE-OUT (Total Folds: {len(patients)}) <<<")
    
    # Precargar UNA sola vez
    preloaded_data = preload_all_patients(patients)
    
    results_table = []

    for test_patient in patients:
        print(f"\n=== Test: {test_patient} (Train: Resto) ===", flush=True)
        
        train_patients = [p for p in patients if p != test_patient]

        try:
            # Reutilizar la función existente, pasándole los datos precargados
            train_loader, val_loader = get_leave_one_out_dataloader(
                train_patients, test_patient, 
                preloaded_data=preloaded_data
            )
            
            model = ChannelFusionCNN()
            trainer = Trainer(model, patience=3)
            trainer.train(train_loader, val_loader, epochs=EPOCHS)
            
            metrics = trainer.evaluate(val_loader)
            metrics['patient'] = test_patient
            results_table.append(metrics)
            
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

        except Exception as e:
             print(f"ERROR FATAL en LOO {test_patient}: {e}", flush=True)

    return results_table


def main():
    print(f"Dispositivo actual: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Info: {torch.cuda.get_device_name(0)}")

    # Detectar pacientes
    eeg_files = list(DATA_PATH.glob("chb*_seizure_EEGwindow_*.npz"))
    # Extraer IDs únicos (chb01, chb02...)
    all_patients = sorted(list(set([f.stem.split("_", 1)[0] for f in eeg_files])))
    
    # Filtrar solo válidos con metadata
    valid_patients = []
    for p in all_patients:
        if (DATA_PATH / f"{p}_seizure_metadata_1.parquet").exists():
            valid_patients.append(p)
    
    print(f"Pacientes detectados ({len(valid_patients)}): {valid_patients}")

    # Ejecutar según modo
    if MODE == 'personalized':
        results = run_personalized_experiment(valid_patients)
        filename = "results_personalized.csv"
    elif MODE == 'leave_one_out':
        results = run_leave_one_out_experiment(valid_patients)
        filename = "results_leave_one_out.csv"
    else:
        print("Modo desconocido.")
        return

    # Guardar resultados
    if results:
        df = pd.DataFrame(results)
        print(f"\n=== RESULTADOS FINALES ({MODE}) ===")
        print(df)
        df.to_csv(filename, index=False)
        print(f"Guardado en {filename}")
    else:
        print("No se generaron resultados.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()