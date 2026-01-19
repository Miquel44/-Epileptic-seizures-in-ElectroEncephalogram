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
import json

from config import *
from data_loader import get_patient_dataloader, get_leave_one_out_dataloader, preload_all_patients
from models import ChannelFusionCNN
from EpilepsyLSTM import EpilepsyLSTM, EpilepsyCNNLSTMSeq, get_default_hyperparameters
from trainer import Trainer

# >>> CONFIGURACIÓN DEL EXPERIMENTO <<<
MODE = 'personalized' 
# MODE = 'leave_one_out'
# MODE = 'prova'

# MODEL = 'SIS1'
MODEL = 'SIS2'

def run_personalized_experiment(patients):
    print(f"\n>>> INICIANDO EXPERIMENTO PERSONALIZADO (Total: {len(patients)}) <<<")
    results_table = []
    all_histories = {}
    all_confusion_matrices = {}

    for patient_id in patients:
        print(f"\n=== Paciente {patient_id} ===", flush=True)
        try:
            # 1. Splits temporales
            train_loader, val_loader = get_patient_dataloader(
                patient_id,
                sequential=True,
                seq_len=10,
                label_mode="last",
                stride=1
            )
            
            # Verificar datos suficientes
            if len(train_loader) == 0: 
                print("SKIP: Datos vacíos.")
                continue

            # 2. Entrenar
            # Seleccionar modelo según configuración
            if MODEL == 'SIS2':
                model = EpilepsyCNNLSTMSeq()
            else:  # SIS1
                model = ChannelFusionCNN()
            
            trainer = Trainer(model, patience=5)
            trainer.train(train_loader, val_loader, epochs=EPOCHS)
            
            # 3. Evaluar
            metrics = trainer.evaluate(val_loader)
            
            all_histories[patient_id] = trainer.history
            all_confusion_matrices[patient_id] = metrics.get('confusion_matrix', [[0,0],[0,0]])
            
            # Preparar métricas para CSV (sin campos grandes)
            metrics_csv = {
                'patient': patient_id,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1']
            }
            results_table.append(metrics_csv)

            # 4. Guardar modelo
            os.makedirs("models", exist_ok=True)
            trainer.save_model(f"models/personalized_{MODEL}_{patient_id}.pth")
            
            # Limpieza
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR FATAL en {patient_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Guardar historiales de entrenamiento
    with open("histories_personalized_sis2.json", 'w') as f:
        json.dump(all_histories, f)
    with open("confusion_matrices_personalized_sis2_1.json", 'w') as f:
        json.dump(all_confusion_matrices, f)
    print("Guardado: histories_personalized.json, confusion_matrices_personalized.json")

    return results_table


def run_leave_one_out_experiment(patients):
    print(f"\n>>> INICIANDO EXPERIMENTO LEAVE-ONE-OUT (Total Folds: {len(patients)}) <<<")
    
    # Precargar UNA sola vez
    preloaded_data = preload_all_patients(patients)
    
    results_table = []
    all_histories = {} 
    all_confusion_matrices = {}

    for test_patient in patients:
        print(f"\n=== Test: {test_patient} (Train: Resto) ===", flush=True)
        
        train_patients = [p for p in patients if p != test_patient]

        try:
            # Reutilizar la función existente, pasándole los datos precargados
            train_loader, val_loader = get_leave_one_out_dataloader(
                train_patients, test_patient, 
                preloaded_data=preloaded_data
            )
            
            # Seleccionar modelo según configuración
            if MODEL == 'SIS2':
                inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
                model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
            else:  # SIS1
                model = ChannelFusionCNN()

            trainer = Trainer(model, patience=3)
            trainer.train(train_loader, val_loader, epochs=EPOCHS)
            
            metrics = trainer.evaluate(val_loader)

            all_histories[test_patient] = trainer.history
            all_confusion_matrices[test_patient] = metrics.get('confusion_matrix', [[0,0],[0,0]])
            
            # Guardar modelo LOO
            os.makedirs("models_80_20", exist_ok=True)
            trainer.save_model(f"models_80_20/loo_{MODEL}_{test_patient}.pth")

            # Preparar métricas para CSV (sin campos grandes)
            metrics_csv = {
                'patient': test_patient,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1']
            }
            results_table.append(metrics_csv)
            
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR FATAL en LOO {test_patient}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Guardar historiales
    with open("histories_leave_one_out.json", 'w') as f:
        json.dump(all_histories, f)
    with open("confusion_matrices_leave_one_out.json", 'w') as f:
        json.dump(all_confusion_matrices, f)
    print("Guardado: histories_leave_one_out.json, confusion_matrices_leave_one_out.json")

    return results_table

def run_single_patient_personalized_experiment(patient_id: str, save: bool = True):
    print(f"\n>>> PRUEBA PERSONALIZADA (1 paciente): {patient_id} <<<", flush=True)

    results_table = []

    # try:
    # 1) Cargar splits del paciente
    train_loader, val_loader = get_patient_dataloader(
        patient_id,
        sequential=True,
        seq_len=5,
        label_mode="last",
        stride=1
    )

    if len(train_loader) == 0:
        print("SKIP: train_loader vacío.")
        return None

    # 2) Entrenar
    model = EpilepsyCNNLSTMSeq()
    trainer = Trainer(model, patience=5)
    trainer.train(train_loader, val_loader, epochs=EPOCHS)

    # 3) Evaluar
    metrics = trainer.evaluate(val_loader)
    metrics['patient'] = patient_id
    results_table.append(metrics)

    # 4) Guardar modelo (opcional)
    if save:
        os.makedirs("models", exist_ok=True)
        trainer.save_model(f"models/personalized_{patient_id}.pth")
        print(f"Modelo guardado en models/personalized_{patient_id}.pth")

    # Limpieza
    del model, trainer, train_loader, val_loader
    torch.cuda.empty_cache()

    return results_table

    # except Exception as e:
    #     print(f"ERROR FATAL en {patient_id}: {e}", flush=True)
    #     return None


def main():
    print(f"Dispositivo actual: {DEVICE}")
    print(f"Modelo seleccionado: {MODEL}")
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
        filename = "results_personalized_sis2_1.csv"
    elif MODE == 'leave_one_out':
        results = run_leave_one_out_experiment(valid_patients)
        filename = "results_leave_one_out_1.csv"
    elif MODE == 'prova':
        results = run_single_patient_personalized_experiment(random.choice(valid_patients), False)
        filename = "results_prova.csv"
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