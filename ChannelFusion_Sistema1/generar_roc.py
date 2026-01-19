"""
Script para generar datos ROC a partir de modelos ya entrenados.
NO requiere re-entrenar, solo carga los modelos y evalúa.
"""
import torch
import numpy as np
import json
from pathlib import Path

from config import DEVICE, DATA_PATH
from models import ChannelFusionCNN
from data_loader import get_patient_dataloader

# Configuración
MODELS_DIR = Path("models_sis1")  # Donde están los .pth
OUTPUT_FILE = "roc_data_personalized.json"


def evaluate_for_roc(model, data_loader):
    """Evalúa modelo y devuelve y_true, y_probs para ROC."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(DEVICE)
            outputs = model(data)
            
            # Probabilidades con softmax
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob de clase 1 (Seizure)

    return all_labels, all_probs


def main():
    print(f"Dispositivo: {DEVICE}")
    
    # Detectar pacientes desde los modelos guardados
    model_files = list(MODELS_DIR.glob("personalized_SIS1_*.pth"))
    patients = [f.stem.split("_")[-1] for f in model_files]
    patients = sorted(set(patients))
    
    print(f"Pacientes encontrados: {len(patients)}")
    
    roc_data = {}
    
    for patient_id in patients:
        print(f"\n=== {patient_id} ===", flush=True)
        
        model_path = MODELS_DIR / f"personalized_SIS1_{patient_id}.pth"
        
        if not model_path.exists():
            print(f"   [SKIP] No existe modelo para {patient_id}")
            continue
        
        try:
            # 1. Cargar datos (mismo split que en entrenamiento)
            _, val_loader = get_patient_dataloader(patient_id)
            
            # 2. Cargar modelo
            model = ChannelFusionCNN()
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)
            
            # 3. Evaluar
            y_true, y_probs = evaluate_for_roc(model, val_loader)
            
            # 4. Guardar
            roc_data[patient_id] = {
                'y_true': [int(x) for x in y_true],      # Convertir a int nativo
                'y_probs': [float(x) for x in y_probs]   # Convertir a float nativo
            }
            
            # Calcular AUC rápido para verificar
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_probs)
            print(f"   AUC: {auc:.4f}")
            
            # Limpiar memoria
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   [ERROR] {e}")
    
    # Guardar JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(roc_data, f)
    
    print(f"\n[OK] Guardado: {OUTPUT_FILE}")
    print(f"     Pacientes procesados: {len(roc_data)}")


if __name__ == "__main__":
    main()