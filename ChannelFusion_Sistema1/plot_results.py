"""
Genera visualizaciones para comparar experimentos:
- Boxplots de métricas
- Curvas de Train/Val Loss (si hay historiales)
- Tabla de métricas
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results():
    """Carga los CSVs de resultados."""
    results = {}
    histories = {}
    conf_matrices = {}
    roc_data = {}
    
    for exp_type in ['personalized_sis2', 'leave_one_out']:
        csv_path = f"results_{exp_type}.csv"
        json_path = f"histories_{exp_type}.json"
        json_cm = f"confusion_matrices_{exp_type}.json"

        # NUEVO: Cargar datos ROC
        roc_path = f"roc_data_{exp_type}.json"
        if Path(roc_path).exists():
            with open(roc_path, 'r') as f:
                roc_data[exp_type] = json.load(f)
            print(f"[OK] Cargado {roc_path}")
        
        if Path(csv_path).exists():
            results[exp_type] = pd.read_csv(csv_path)
            print(f"[OK] Cargado {csv_path} ({len(results[exp_type])} pacientes)")
        
        if Path(json_path).exists():
            with open(json_path, 'r') as f:
                histories[exp_type] = json.load(f)
            print(f"[OK] Cargado {json_path}")

        if Path(json_cm).exists():
            with open(json_cm, 'r') as f:
                conf_matrices[exp_type] = json.load(f)
            print(f"[OK] Cargado {json_cm}")
    
    return results, histories, conf_matrices, roc_data

def plot_roc_curves(roc_data, output_path="roc_curves_sis2.png"):
    """Genera curvas ROC para cada paciente."""
    
    for exp_type, patients_data in roc_data.items():
        n_patients = len(patients_data)
        
        if n_patients == 0:
            continue
        
        n_cols = min(6, n_patients)
        n_rows = int(np.ceil(n_patients / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
        
        if n_patients == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        fig.suptitle(f'Curvas ROC - {exp_label}', fontsize=16, fontweight='bold', y=1.02)
        
        all_aucs = []
        
        for idx, (patient_id, data) in enumerate(sorted(patients_data.items())):
            y_true = np.array(data['y_true'])
            y_probs = np.array(data['y_probs'])
            
            ax = axes[idx]
            
            # Verificar que hay ambas clases
            if len(np.unique(y_true)) < 2:
                ax.text(0.5, 0.5, 'N/A\n(solo 1 clase)', ha='center', va='center', fontsize=10)
                ax.set_title(f'{patient_id}', fontsize=10, fontweight='bold')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                continue
            
            # Calcular ROC
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            all_aucs.append(roc_auc)
            
            ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('FPR', fontsize=8)
            ax.set_ylabel('TPR', fontsize=8)
            ax.set_title(f'{patient_id}', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=7)
            ax.tick_params(axis='both', labelsize=7)
        
        # Ocultar ejes vacíos
        for idx in range(n_patients, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = output_path.replace('.png', f'_{exp_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calcular media solo de AUCs válidos
        mean_auc = np.mean(all_aucs) if all_aucs else float('nan')
        print(f"[OK] Guardado: {save_path} (Media AUC: {mean_auc:.4f}, {len(all_aucs)}/{n_patients} válidos)")


def plot_roc_aggregated(roc_data, output_path="roc_aggregated_sis2.png"):
    """Curva ROC agregada (todos los pacientes juntos) + comparación."""
    
    n_plots = len(roc_data)
    if n_plots == 0:
        print("[SKIP] No hay datos ROC para agregar")
        return
    
    # Si hay 2 experimentos: 3 subplots (uno por cada + comparación)
    # Si hay 1 experimento: 1 subplot
    n_cols = n_plots + 1 if n_plots >= 2 else 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    
    # Asegurar que axes sea siempre una lista
    if n_cols == 1:
        axes = [axes]
    
    colors = {'personalized': 'steelblue', 'leave_one_out': 'coral'}
    labels = {'personalized': 'Personalizado', 'leave_one_out': 'Leave-One-Out'}
    
    all_results = {}
    
    for idx, (exp_type, patients_data) in enumerate(roc_data.items()):
        # Agregar todos los datos
        y_true_all = []
        y_probs_all = []
        
        for patient_id, data in patients_data.items():
            y_true_all.extend(data['y_true'])
            y_probs_all.extend(data['y_probs'])
        
        y_true_all = np.array(y_true_all)
        y_probs_all = np.array(y_probs_all)
        
        # Verificar que hay ambas clases
        if len(np.unique(y_true_all)) < 2:
            print(f"[WARN] {exp_type}: Solo una clase en datos agregados, saltando ROC")
            continue
        
        # Calcular ROC agregada
        fpr, tpr, _ = roc_curve(y_true_all, y_probs_all)
        roc_auc = auc(fpr, tpr)
        
        all_results[exp_type] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        # Plot individual
        ax = axes[idx]
        ax.plot(fpr, tpr, color=colors.get(exp_type, 'steelblue'), lw=2, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Agregada - {labels.get(exp_type, exp_type)}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Plot comparativo (último subplot) solo si hay 2+ experimentos
    if len(all_results) >= 2 and n_cols > 1:
        ax = axes[-1]
        for exp_type, res in all_results.items():
            ax.plot(res['fpr'], res['tpr'], color=colors.get(exp_type, 'gray'), lw=2, 
                   label=f'{labels.get(exp_type, exp_type)} (AUC={res["auc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Comparación ROC', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")

def plot_confusion_matrix_single(cm, patient_id, exp_label, ax=None, normalize=True):
    """Plotea una matriz de confusión individual."""
    cm = np.array(cm)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'],
                cbar=False)
    
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f'{patient_id}', fontsize=11, fontweight='bold')
    
    # Añadir conteos absolutos como texto secundario
    if normalize:
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, f'(n={cm[i,j]})', 
                       ha='center', va='center', fontsize=7, color='gray')

def plot_all_confusion_matrices(conf_matrices, output_path="confusion_matrices_sis2.png"):
    """Genera grid de matrices de confusión para todos los pacientes."""
    
    for exp_type, cm_dict in conf_matrices.items():
        n_patients = len(cm_dict)
        
        if n_patients == 0:
            continue
        
        n_cols = min(6, n_patients)
        n_rows = int(np.ceil(n_patients / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
        
        if n_patients == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        fig.suptitle(f'Matrices de Confusión - {exp_label}', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        for idx, (patient_id, cm) in enumerate(sorted(cm_dict.items())):
            plot_confusion_matrix_single(cm, patient_id, exp_label, ax=axes[idx])
        
        # Ocultar ejes vacíos
        for idx in range(n_patients, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = output_path.replace('.png', f'_{exp_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Guardado: {save_path}")

def plot_aggregated_confusion_matrix(conf_matrices, output_path="confusion_matrix_aggregated_sis2.png"):
    """Genera matriz de confusión agregada (suma de todos los pacientes)."""
    
    fig, axes = plt.subplots(1, len(conf_matrices), figsize=(6 * len(conf_matrices), 5))
    
    if len(conf_matrices) == 1:
        axes = [axes]
    
    for idx, (exp_type, cm_dict) in enumerate(conf_matrices.items()):
        # Sumar todas las matrices
        aggregated_cm = np.zeros((2, 2), dtype=int)
        for patient_id, cm in cm_dict.items():
            aggregated_cm += np.array(cm)
        
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        
        # Calcular métricas desde la matriz agregada
        tn, fp, fn, tp = aggregated_cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Normalizar para visualización
        cm_normalized = aggregated_cm.astype('float') / aggregated_cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Seizure'],
                    yticklabels=['Normal', 'Seizure'])
        
        # Añadir anotaciones personalizadas
        for i in range(2):
            for j in range(2):
                text = f'{cm_normalized[i,j]:.1%}\n({aggregated_cm[i,j]:,})'
                ax.text(j + 0.5, i + 0.5, text, 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        ax.set_title(f'{exp_label}\n(Precision={precision:.3f}, Recall={recall:.3f}, Spec={specificity:.3f})', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")

def plot_confusion_comparison(conf_matrices, output_path="confusion_comparison_sis2.png"):
    """Compara matrices de confusión normalizadas lado a lado."""
    
    if len(conf_matrices) < 2:
        print("[SKIP] Necesitas ambos experimentos para comparar")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (exp_type, cm_dict) in enumerate(conf_matrices.items()):
        # Agregar todas las matrices
        aggregated_cm = np.zeros((2, 2), dtype=int)
        for cm in cm_dict.values():
            aggregated_cm += np.array(cm)
        
        # Normalizar por filas (para mostrar recall por clase)
        cm_normalized = aggregated_cm.astype('float') / aggregated_cm.sum(axis=1)[:, np.newaxis]
        
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        
        ax = axes[idx]
        im = sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax,
                        xticklabels=['Normal', 'Seizure'],
                        yticklabels=['Normal', 'Seizure'],
                        vmin=0, vmax=1, cbar=True)
        
        # Anotaciones con porcentaje y conteo
        for i in range(2):
            for j in range(2):
                text = f'{cm_normalized[i,j]:.1%}\n({aggregated_cm[i,j]:,})'
                color = 'white' if cm_normalized[i,j] > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text, 
                       ha='center', va='center', fontsize=11, 
                       fontweight='bold', color=color)
        
        ax.set_xlabel('Predicción', fontsize=12)
        ax.set_ylabel('Real', fontsize=12)
        ax.set_title(f'{exp_label}', fontsize=14, fontweight='bold')
    
    plt.suptitle('Comparación de Matrices de Confusión Agregadas', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")

def plot_boxplots(results, output_path="boxplots_comparison_sis2.png"):
    """Genera boxplots comparando Accuracy y F1 entre experimentos."""
    
    if not results:
        print("[SKIP] No hay resultados para boxplots")
        return
    
    # Preparar datos para seaborn
    data_list = []
    for exp_type, df in results.items():
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        for _, row in df.iterrows():
            data_list.append({
                'Experimento': exp_label,
                'Paciente': row['patient'],
                'Accuracy': row['accuracy'],
                'F1-Score': row['f1']
            })
    
    plot_df = pd.DataFrame(data_list)

    # Calcular límites automáticos con margen
    def get_ylim(values, margin_pct=0.1):
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val
        margin = max(range_val * margin_pct, 0.02)  # Mínimo 2% de margen
        y_min = max(0, min_val - margin)  # No bajar de 0
        y_max = min(1.05, max_val + margin)  # No subir de 1.05
        return (y_min, y_max)
    
    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot Accuracy
    sns.boxplot(data=plot_df, x='Experimento', y='Accuracy', ax=axes[0], palette='Set2')
    sns.stripplot(data=plot_df, x='Experimento', y='Accuracy', ax=axes[0], 
                  color='black', alpha=0.5, size=4)
    axes[0].set_title('Distribución de Accuracy por Experimento', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim(get_ylim(plot_df['Accuracy']))
    
    # Añadir media como línea horizontal
    for i, exp in enumerate(plot_df['Experimento'].unique()):
        mean_val = plot_df[plot_df['Experimento'] == exp]['Accuracy'].mean()
        axes[0].axhline(y=mean_val, xmin=i/2, xmax=(i+1)/2, color='red', linestyle='--', alpha=0.7)
        axes[0].text(i, mean_val + 0.02, f'μ={mean_val:.3f}', ha='center', fontsize=9, color='red')
    
    # Boxplot F1-Score
    sns.boxplot(data=plot_df, x='Experimento', y='F1-Score', ax=axes[1], palette='Set2')
    sns.stripplot(data=plot_df, x='Experimento', y='F1-Score', ax=axes[1], 
                  color='black', alpha=0.5, size=4)
    axes[1].set_title('Distribución de F1-Score por Experimento', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_ylim(get_ylim(plot_df['F1-Score']))
    
    for i, exp in enumerate(plot_df['Experimento'].unique()):
        mean_val = plot_df[plot_df['Experimento'] == exp]['F1-Score'].mean()
        axes[1].axhline(y=mean_val, xmin=i/2, xmax=(i+1)/2, color='red', linestyle='--', alpha=0.7)
        axes[1].text(i, mean_val + 0.02, f'μ={mean_val:.3f}', ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")


def plot_single_experiment_boxplot(results, exp_type, output_path=None):
    """Genera boxplot para un solo experimento."""
    
    if exp_type not in results:
        print(f"[SKIP] No hay resultados para {exp_type}")
        return
    
    df = results[exp_type]
    exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
    
    if output_path is None:
        output_path = f"boxplot_{exp_type}.png"

    # Calcular límites automáticos con margen
    def get_ylim(values, margin_pct=0.1):
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val
        margin = max(range_val * margin_pct, 0.02)
        y_min = max(0, min_val - margin)
        y_max = min(1.05, max_val + margin)
        return (y_min, y_max)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    acc_ylim = get_ylim(df['accuracy'])
    sns.boxplot(y=df['accuracy'], ax=axes[0], color='steelblue')
    sns.stripplot(y=df['accuracy'], ax=axes[0], color='black', alpha=0.5, size=6)
    axes[0].set_title(f'Accuracy - {exp_label}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim(acc_ylim)
    mean_acc = df['accuracy'].mean()
    axes[0].axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7)
    axes[0].text(0.1, mean_acc + (acc_ylim[1] - acc_ylim[0]) * 0.03, 
                 f'Media: {mean_acc:.4f}', fontsize=10, color='red')
    
    # F1
    f1_ylim = get_ylim(df['f1'])
    sns.boxplot(y=df['f1'], ax=axes[1], color='coral')
    sns.stripplot(y=df['f1'], ax=axes[1], color='black', alpha=0.5, size=6)
    axes[1].set_title(f'F1-Score - {exp_label}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_ylim(f1_ylim)
    mean_f1 = df['f1'].mean()
    axes[1].axhline(y=mean_f1, color='red', linestyle='--', alpha=0.7)
    axes[1].text(0.1, mean_f1 + (f1_ylim[1] - f1_ylim[0]) * 0.03, 
                 f'Media: {mean_f1:.4f}', fontsize=10, color='red')
    
    plt.suptitle(f'Métricas por Paciente - {exp_label}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")


def plot_loss_curves(histories, output_path="loss_curves_sis2.png"):
    """Genera curvas de Train/Val Loss para cada paciente."""
    
    for exp_type, hist_dict in histories.items():
        n_patients = len(hist_dict)
        
        if n_patients == 0:
            continue
        
        n_cols = 6
        n_rows = int(np.ceil(n_patients / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
        axes = axes.flatten() if n_patients > 1 else [axes]
        
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        fig.suptitle(f'Curvas de Loss - {exp_label}', fontsize=16, fontweight='bold', y=1.02)
        
        for idx, (patient_id, history) in enumerate(sorted(hist_dict.items())):
            ax = axes[idx]
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
            
            ax.set_title(patient_id, fontsize=10, fontweight='bold')
            ax.set_xlabel('Época', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
        
        for idx in range(n_patients, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = output_path.replace('.png', f'_{exp_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Guardado: {save_path}")


def generate_metrics_table(results, output_path="metrics_table_sis2.csv"):
    """Genera tabla de métricas con estadísticas."""
    
    summary_data = []
    
    for exp_type, df in results.items():
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        
        summary_data.append({
            'Experimento': exp_label,
            'Métrica': 'Accuracy',
            'Media': df['accuracy'].mean(),
            'Std': df['accuracy'].std(),
            'Min': df['accuracy'].min(),
            'Max': df['accuracy'].max(),
            'Mediana': df['accuracy'].median()
        })
        
        summary_data.append({
            'Experimento': exp_label,
            'Métrica': 'F1-Score',
            'Media': df['f1'].mean(),
            'Std': df['f1'].std(),
            'Min': df['f1'].min(),
            'Max': df['f1'].max(),
            'Mediana': df['f1'].median()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Formatear números
    for col in ['Media', 'Std', 'Min', 'Max', 'Mediana']:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}")
    
    summary_df.to_csv(output_path, index=False)
    print(f"[OK] Guardado: {output_path}")
    
    print("\n" + "="*80)
    print("TABLA DE MÉTRICAS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")
    
    return summary_df


def generate_per_patient_table(results, output_path="metrics_per_patient_sis2.csv"):
    """Genera tabla detallada por paciente."""
    
    all_data = []
    
    for exp_type, df in results.items():
        exp_label = "Personalizado" if exp_type == "personalized" else "Leave-One-Out"
        for _, row in df.iterrows():
            all_data.append({
                'Experimento': exp_label,
                'Paciente': row['patient'],
                'Accuracy': f"{row['accuracy']:.4f}",
                'F1-Score': f"{row['f1']:.4f}"
            })
    
    detail_df = pd.DataFrame(all_data)
    detail_df.to_csv(output_path, index=False)
    print(f"[OK] Guardado: {output_path}")
    
    return detail_df


def plot_patient_comparison(results, output_path="patient_comparison_sis2.png"):
    """Barras comparando cada paciente entre experimentos."""
    
    if len(results) < 2:
        print("[SKIP] Necesitas ambos experimentos para comparar por paciente")
        return
    
    df_pers = results.get('personalized', pd.DataFrame())
    df_loo = results.get('leave_one_out', pd.DataFrame())
    
    if df_pers.empty or df_loo.empty:
        return
    
    merged = df_pers.merge(df_loo, on='patient', suffixes=('_pers', '_loo'))
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    x = np.arange(len(merged))
    width = 0.35
    
    # Accuracy
    axes[0].bar(x - width/2, merged['accuracy_pers'], width, label='Personalizado', color='steelblue')
    axes[0].bar(x + width/2, merged['accuracy_loo'], width, label='Leave-One-Out', color='coral')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Comparación de Accuracy por Paciente', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(merged['patient'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=merged['accuracy_pers'].mean(), color='steelblue', linestyle='--', alpha=0.7, label=f'Media Pers: {merged["accuracy_pers"].mean():.3f}')
    axes[0].axhline(y=merged['accuracy_loo'].mean(), color='coral', linestyle='--', alpha=0.7)
    
    # F1-Score
    axes[1].bar(x - width/2, merged['f1_pers'], width, label='Personalizado', color='steelblue')
    axes[1].bar(x + width/2, merged['f1_loo'], width, label='Leave-One-Out', color='coral')
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Comparación de F1-Score por Paciente', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(merged['patient'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(y=merged['f1_pers'].mean(), color='steelblue', linestyle='--', alpha=0.7)
    axes[1].axhline(y=merged['f1_loo'].mean(), color='coral', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Guardado: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERANDO VISUALIZACIONES Y MÉTRICAS")
    print("="*60 + "\n")
    
    results, histories, conf_matrices, roc_data = load_results()
    
    if not results:
        print("[ERROR] No se encontraron archivos CSV de resultados.")
        print("Archivos esperados: results_personalized.csv, results_leave_one_out.csv")
        exit(1)
    
    print("\n--- Generando gráficos ---\n")
    
    # 1. Boxplots individuales por experimento
    for exp_type in results.keys():
        plot_single_experiment_boxplot(results, exp_type)
    
    # 2. Boxplots comparativos (si hay ambos)
    if len(results) >= 2:
        plot_boxplots(results)
    
    # 3. Curvas de Loss (si hay historiales)
    if histories:
        plot_loss_curves(histories)
    else:
        print("[INFO] No hay historiales JSON. Saltando curvas de loss.")

    # 4. Matrices de confusión
    if conf_matrices:
        plot_all_confusion_matrices(conf_matrices)
        plot_aggregated_confusion_matrix(conf_matrices)
        if len(conf_matrices) >= 2:
            plot_confusion_comparison(conf_matrices)
    else:
        print("[INFO] No hay matrices de confusión. Saltando.")

    if roc_data:
        plot_roc_curves(roc_data)
        plot_roc_aggregated(roc_data)
    else:
        print("[INFO] No hay datos ROC. Saltando curvas ROC.")
    
    # 5. Tablas
    generate_metrics_table(results)
    generate_per_patient_table(results)
    
    # 6. Comparación por paciente
    if len(results) == 2:
        plot_patient_comparison(results)
    
    print("\n" + "="*60)
    print("¡COMPLETADO!")
    print("="*60)