"""
Script para generar matrices de confusión y boxplots desde el archivo de log.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def parse_log_file(log_path):
    """Parsea el log y extrae métricas y matrices de confusión."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Buscar todos los bloques de pacientes
    patient_pattern = r"=== Test: (\w+) \(Train: Resto\) ===(.*?)(?==== Test:|=== RESULTADOS FINALES)"
    blocks = re.findall(patient_pattern, content, re.DOTALL)
    
    results = {}
    
    for patient_id, block in blocks:
        # Extraer métricas del reporte de clasificación
        # Buscar líneas de precision/recall/f1
        normal_match = re.search(r"Normal\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", block)
        seizure_match = re.search(r"Seizure\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", block)
        accuracy_match = re.search(r"accuracy\s+([\d.]+)\s+(\d+)", block)
        
        if normal_match and seizure_match and accuracy_match:
            normal_support = int(normal_match.group(4))
            seizure_support = int(seizure_match.group(4))
            
            # Calcular matriz de confusión a partir de precision, recall y support
            # Normal: precision, recall, f1, support
            normal_precision = float(normal_match.group(1))
            normal_recall = float(normal_match.group(2))
            normal_f1 = float(normal_match.group(3))
            
            # Seizure: precision, recall, f1, support
            seizure_precision = float(seizure_match.group(1))
            seizure_recall = float(seizure_match.group(2))
            seizure_f1 = float(seizure_match.group(3))
            
            accuracy = float(accuracy_match.group(1))
            total = int(accuracy_match.group(2))
            
            # Reconstruir matriz de confusión
            # TN = Normal correctos = recall_normal * support_normal
            # FP = Normal incorrectos = support_normal - TN
            # TP = Seizure correctos = recall_seizure * support_seizure
            # FN = Seizure incorrectos = support_seizure - TP
            
            tn = int(round(normal_recall * normal_support))
            fp = normal_support - tn
            tp = int(round(seizure_recall * seizure_support))
            fn = seizure_support - tp
            
            # Ajustar: en sklearn la matriz es [[TN, FP], [FN, TP]]
            # pero aquí Normal=0, Seizure=1
            # Matriz: [[pred_0_real_0, pred_1_real_0], [pred_0_real_1, pred_1_real_1]]
            # = [[TN, FP], [FN, TP]]
            
            results[patient_id] = {
                'accuracy': accuracy,
                'f1_normal': normal_f1,
                'f1_seizure': seizure_f1,
                'precision_normal': normal_precision,
                'precision_seizure': seizure_precision,
                'recall_normal': normal_recall,
                'recall_seizure': seizure_recall,
                'support_normal': normal_support,
                'support_seizure': seizure_support,
                'confusion_matrix': [[tn, fp], [fn, tp]]
            }
    
    return results


def plot_confusion_matrices(results, output_path="confusion_matrices_loo_from_log.png"):
    """Genera grid de matrices de confusión bonitas (normalizadas por fila)."""

    n_patients = len(results)
    n_cols = min(6, n_patients)
    n_rows = int(np.ceil(n_patients / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    fig.suptitle(
        'Matrices de Confusión - Leave-One-Out (CNN)\n(Normalizadas por clase real)',
        fontsize=15,
        fontweight='bold',
        y=1.03
    )

    for idx, (patient_id, data) in enumerate(sorted(results.items())):
        ax = axes[idx]
        cm = np.array(data['confusion_matrix'])

        # Normalización POR FILAS (clase real)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm,
            ax=ax,
            cmap='Blues',
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=['Normal', 'Seizure'],
            yticklabels=['Normal', 'Seizure']
        )

        # Anotaciones: porcentaje por fila + conteo absoluto
        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j] * 100
                count = cm[i, j]
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f'{pct:.1f}%\n(n={count})',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color=color
                )

        ax.set_title(
            f'{patient_id}\nAcc: {data["accuracy"]:.1%}',
            fontsize=10,
            fontweight='bold'
        )
        ax.set_xlabel('Predicho', fontsize=9)
        ax.set_ylabel('Real', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

    # Ocultar ejes vacíos
    for idx in range(n_patients, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")


def plot_boxplots(results, output_path="boxplots_loo_from_log.png"):
    """Genera boxplots de métricas."""
    
    # Crear DataFrame con métricas
    data = []
    for patient_id, metrics in results.items():
        data.append({
            'patient': patient_id,
            'Accuracy': metrics['accuracy'],
            'F1 Normal': metrics['f1_normal'],
            'F1 Seizure': metrics['f1_seizure'],
            'Precision Seizure': metrics['precision_seizure'],
            'Recall Seizure': metrics['recall_seizure']
        })
    
    df = pd.DataFrame(data)
    
    # Calcular F1 macro como promedio
    df['F1 Macro'] = (df['F1 Normal'] + df['F1 Seizure']) / 2
    
    # Preparar datos para boxplot
    metrics_cols = ['Accuracy', 'F1 Macro', 'F1 Seizure', 'Precision Seizure', 'Recall Seizure']
    
    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(3.5 * len(metrics_cols), 5))
    
    fig.suptitle('Distribución de Métricas - Leave-One-Out (CNN)', fontsize=14, fontweight='bold', y=1.02)
    
    colors = sns.color_palette("husl", len(metrics_cols))
    
    for idx, metric in enumerate(metrics_cols):
        ax = axes[idx]
        values = df[metric].values
        
        bp = ax.boxplot(values, patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(colors[idx])
        bp['boxes'][0].set_alpha(0.7)
        
        # Añadir puntos individuales
        ax.scatter(np.ones(len(values)), values, alpha=0.5, color='darkblue', s=30, zorder=3)
        
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{metric}\n(μ={np.mean(values):.3f}, σ={np.std(values):.3f})', 
                     fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")
    
    # Imprimir resumen
    print("\n=== RESUMEN ESTADÍSTICO ===")
    for metric in metrics_cols:
        values = df[metric].values
        print(f"{metric}: μ={np.mean(values):.4f}, σ={np.std(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}")


def plot_aggregated_confusion_matrix(results, output_path="confusion_matrix_aggregated_loo.png"):
    """Genera matriz de confusión agregada."""
    
    total_cm = np.zeros((2, 2), dtype=int)
    
    for patient_id, data in results.items():
        total_cm += np.array(data['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calcular porcentajes
    total = total_cm.sum()
    cm_pct = total_cm.astype(float) / total * 100
    
    # Crear anotaciones
    annot = np.array([[f'{total_cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)' 
                       for j in range(2)] for i in range(2)])
    
    sns.heatmap(total_cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'],
                annot_kws={'size': 14})
    
    # Calcular métricas globales
    tn, fp, fn, tp = total_cm.ravel()
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    ax.set_title(f'Matriz de Confusión Agregada - Leave-One-Out (CNN)\n'
                 f'Acc: {accuracy:.2%} | Prec: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}\n'
                 f'Total: {total:,} muestras', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicho', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")


if __name__ == "__main__":
    # Ruta al log
    log_path = "ChannelFusion_Sistema1/logs/ChannelFusion_99879.out"
    
    print(f"Parseando log: {log_path}")
    results = parse_log_file(log_path)
    print(f"Encontrados {len(results)} pacientes")
    
    if results:
        plot_confusion_matrices(results, "confusion_matrices_loo_from_log.png")
        plot_boxplots(results, "boxplots_loo_from_log.png")
        plot_aggregated_confusion_matrix(results, "confusion_matrix_aggregated_loo.png")
        
        # Guardar CSV con métricas
        df = pd.DataFrame([
            {'patient': p, 'accuracy': d['accuracy'], 
             'f1_seizure': d['f1_seizure'], 
             'recall_seizure': d['recall_seizure'],
             'precision_seizure': d['precision_seizure']}
            for p, d in sorted(results.items())
        ])
        df.to_csv('metrics_loo_from_log.csv', index=False)
        print("[OK] Guardado: metrics_loo_from_log.csv")