import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_experiments():
    file_pers = "results_personalized.csv"
    file_loo = "results_leave_one_out.csv"

    if not os.path.exists(file_pers) or not os.path.exists(file_loo):
        print("Faltan CSVs. Ejecuta minimain.py en ambos modos.")
        return

    df_p = pd.read_csv(file_pers)
    df_l = pd.read_csv(file_loo)

    # Merge por paciente
    # df_p tiene [patient, accuracy, f1]
    # df_l tiene [patient, accuracy, f1]
    df = pd.merge(df_p, df_l, on='patient', suffixes=('_pers', '_loo'))
    
    # Calcular Gap
    df['F1_Gap'] = df['f1_pers'] - df['f1_loo']
    df = df.sort_values('F1_Gap', ascending=False)

    print("\n=== COMPARATIVA ===")
    print(df[['patient', 'f1_pers', 'f1_loo', 'F1_Gap']].head(24))

    # Plot
    plt.figure(figsize=(15, 8))
    X = np.arange(len(df))
    w = 0.35
    
    plt.bar(X - w/2, df['f1_pers'], w, label='Personalized', color='#1f77b4')
    plt.bar(X + w/2, df['f1_loo'], w, label='Gen. (LOO)', color='#ff7f0e')
    
    plt.xticks(X, df['patient'], rotation=45)
    plt.ylabel('F1 Score')
    plt.title('Comparación: Modelo Específico vs Generalista')
    plt.legend()
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig("analysis_plot.png")
    print("Gráfica guardada en 'analysis_plot.png'")

if __name__ == "__main__":
    compare_experiments()