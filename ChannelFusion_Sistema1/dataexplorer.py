"""
DataExplorer - Visualización de datos EEG para Channel Fusion CNN
Organizado en secciones: Carga, Visualización básica, Estadísticas, Kernels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
from scipy.ndimage import convolve

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DATA_PATH = Path(
    r"C:\Users\mique\OneDrive\Escritorio\Proyectos\-Epileptic-seizures-in-ElectroEncephalogram\Data\input\input"
)
MODEL_PATH = "channel_fusion_model_4patients.pth"


def calcular_duracion_paciente(signals, fs: int = 128):
    """
    Calcula la duración total de todas las ventanas de un paciente.

    Args:
        signals: Array de forma (N_ventanas, canales, muestras)
        fs: Frecuencia de muestreo en Hz (default 256 Hz para CHB-MIT)
    """
    n_ventanas = signals.shape[0]
    muestras_por_ventana = signals.shape[2]

    # Duración de una ventana
    duracion_ventana_seg = muestras_por_ventana / fs

    # Duración total (asumiendo ventanas sin solapamiento)
    duracion_total_seg = n_ventanas * duracion_ventana_seg
    duracion_total_min = duracion_total_seg / 60
    duracion_total_horas = duracion_total_min / 60

    print(f"\n{'=' * 50}")
    print(f"DURACIÓN TOTAL DEL PACIENTE")
    print(f"{'=' * 50}")
    print(f"  Frecuencia muestreo: {fs} Hz")
    print(f"  Ventanas: {n_ventanas}")
    print(f"  Muestras/ventana: {muestras_por_ventana}")
    print(f"  Duración/ventana: {duracion_ventana_seg:.3f} seg ({duracion_ventana_seg * 1000:.1f} ms)")
    print(f"{'=' * 50}")
    print(f"  DURACIÓN TOTAL: {duracion_total_seg:.1f} seg")
    print(f"                  {duracion_total_min:.2f} minutos")
    print(f"                  {duracion_total_horas:.3f} horas")
    print(f"{'=' * 50}")

    return {
        'ventanas': n_ventanas,
        'muestras_ventana': muestras_por_ventana,
        'duracion_ventana_seg': duracion_ventana_seg,
        'duracion_total_seg': duracion_total_seg,
        'duracion_total_min': duracion_total_min,
        'duracion_total_horas': duracion_total_horas
    }
# =============================================================================
# CARGA DE DATOS
# =============================================================================

def cargar_datos(patient_id: str = "chb01"):
    """Carga datos EEG y metadata de un paciente."""
    eeg = np.load(DATA_PATH / f"{patient_id}_seizure_EEGwindow_1.npz", allow_pickle=True)
    metadata = pd.read_parquet(DATA_PATH / f"{patient_id}_seizure_metadata_1.parquet")

    signals = np.array(eeg['EEG_win'], dtype=np.float32)
    labels = metadata['class'].values

    print(f"Datos cargados: {signals.shape}")
    print(f"  - Ventanas: {signals.shape[0]}")
    print(f"  - Canales: {signals.shape[1]}")
    print(f"  - Muestras: {signals.shape[2]}")
    print(f"  - Crisis: {labels.sum()} | Normal: {(labels == 0).sum()}")

    return signals, labels


# =============================================================================
# VISUALIZACIÓN PRINCIPAL: QUÉ CAPTURA EL KERNEL (3x5)
# =============================================================================
def visualizar_kernel_captura(signals, labels, idx=None, pos_canal=10, pos_tiempo=60):
    """
    Visualización clara de qué región captura un kernel (3,5).
    Muestra exactamente los 3 canales y 5 muestras temporales.
    """
    if idx is None:
        idx = np.where(labels == 1)[0][0]  # Primera crisis

    window = signals[idx]
    label_str = 'CRISIS' if labels[idx] == 1 else 'NORMAL'

    fig = plt.figure(figsize=(16, 5))

    # --- Panel 1: Imagen completa con kernel marcado ---
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(window, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_title(f'Señal EEG completa ({label_str})\n21 canales × 128 muestras', fontsize=11)
    ax1.set_xlabel('Tiempo (muestras)')
    ax1.set_ylabel('Canal EEG')
    plt.colorbar(im, ax=ax1, label='Amplitud (μV)')

    # Dibujar rectángulo del kernel
    rect = patches.Rectangle(
        (pos_tiempo - 0.5, pos_canal - 0.5), 5, 3,
        linewidth=3, edgecolor='lime', facecolor='none'
    )
    ax1.add_patch(rect)
    ax1.annotate(
        'Kernel\n(3×5)', xy=(pos_tiempo + 2.5, pos_canal - 1),
        fontsize=10, color='lime', ha='center', fontweight='bold'
    )

    # --- Panel 2: Zoom en la región del kernel ---
    ax2 = fig.add_subplot(1, 3, 2)

    region = window[pos_canal:pos_canal + 3, pos_tiempo:pos_tiempo + 5]

    im2 = ax2.imshow(region, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_title('ZOOM: Lo que "ve" el kernel (3×5)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('← 5 muestras temporales (~20ms) →')
    ax2.set_ylabel('← 3 canales →')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels([f't{i}' for i in range(5)])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([f'Ch{pos_canal + i}' for i in range(3)])
    plt.colorbar(im2, ax=ax2, label='μV')

    for i in range(3):
        for j in range(5):
            ax2.text(j, i, f'{region[i, j]:.0f}', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

    # --- Panel 3: Vista de señales con región marcada ---
    ax3 = fig.add_subplot(1, 3, 3)

    colores = ['#e41a1c', '#377eb8', '#4daf4a']
    for i, ch in enumerate(range(pos_canal, pos_canal + 3)):
        señal = window[ch]
        offset = i * 150
        ax3.plot(señal + offset, color=colores[i], alpha=0.7, label=f'Canal {ch}')
        ax3.axvspan(pos_tiempo, pos_tiempo + 5, alpha=0.3, color='yellow')
        ax3.scatter(
            range(pos_tiempo, pos_tiempo + 5),
            señal[pos_tiempo:pos_tiempo + 5] + offset,
            color=colores[i], s=50, zorder=5, edgecolor='black'
        )

    ax3.axvspan(pos_tiempo, pos_tiempo + 5, alpha=0.2, color='lime', label='Región kernel')
    ax3.set_title('Vista tradicional: 3 canales capturados', fontsize=11)
    ax3.set_xlabel('Tiempo (muestras)')
    ax3.set_ylabel('Amplitud + offset')
    ax3.legend(loc='upper right')

    plt.suptitle('¿Qué captura Conv2d(kernel_size=(3,5))?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kernel_captura_explicacion.png', dpi=150, bbox_inches='tight')
    plt.show()

# =============================================================================
# VISUALIZACIÓN BÁSICA
# =============================================================================

def plot_channel_fusion_input(signals, labels, idx_normal=0, idx_seizure=None):
    """Visualiza entrada normal vs crisis como imágenes 2D."""
    if idx_seizure is None:
        seizure_idx = np.where(labels == 1)[0]
        idx_seizure = seizure_idx[0] if len(seizure_idx) > 0 else idx_normal

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (idx, titulo) in enumerate([(idx_normal, 'NORMAL'), (idx_seizure, 'CRISIS')]):
        window = signals[idx]

        im = axes[row, 0].imshow(window, aspect='auto', cmap='RdBu_r')
        axes[row, 0].set_title(f'{titulo} (idx={idx})')
        axes[row, 0].set_xlabel('Tiempo')
        axes[row, 0].set_ylabel('Canal')
        plt.colorbar(im, ax=axes[row, 0])

        for ch in range(min(5, window.shape[0])):
            axes[row, 1].plot(window[ch] + ch * 100, alpha=0.7, label=f'Ch {ch}')
        axes[row, 1].set_title(f'Vista señales - {titulo}')
        axes[row, 1].legend(loc='upper right', fontsize=8)

    plt.suptitle('EEG como "imagen 2D" para CNN', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('channel_fusion_visualization.png', dpi=150)
    plt.show()


# =============================================================================
# ESTADÍSTICAS
# =============================================================================

def comparar_estadisticas(signals, labels):
    """Compara distribuciones normal vs crisis."""
    normal = signals[labels == 0]
    crisis = signals[labels == 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(normal.flatten(), bins=100, alpha=0.5, label='Normal', density=True)
    axes[0].hist(crisis.flatten(), bins=100, alpha=0.5, label='Crisis', density=True)
    axes[0].set_title('Distribución de amplitudes')
    axes[0].legend()

    var_normal = normal.var(axis=2).mean(axis=0)
    var_crisis = crisis.var(axis=2).mean(axis=0)
    x = np.arange(21)
    axes[1].bar(x - 0.2, var_normal, 0.4, alpha=0.7, label='Normal')
    axes[1].bar(x + 0.2, var_crisis, 0.4, alpha=0.7, label='Crisis')
    axes[1].set_title('Varianza por canal')
    axes[1].legend()

    max_normal = np.abs(normal).max(axis=(1, 2))
    max_crisis = np.abs(crisis).max(axis=(1, 2))
    axes[2].boxplot([max_normal, max_crisis], labels=['Normal', 'Crisis'])
    axes[2].set_title('Amplitud máxima')

    plt.tight_layout()
    plt.savefig('estadisticas_normal_vs_crisis.png', dpi=150)
    plt.show()

    print(f"\nEstadísticas:")
    print(f"  Normal - Media: {normal.mean():.2f}, Std: {normal.std():.2f}")
    print(f"  Crisis - Media: {crisis.mean():.2f}, Std: {crisis.std():.2f}")


# =============================================================================
# VISUALIZACIÓN DE KERNELS ENTRENADOS
# =============================================================================

def visualizar_kernels_modelo(model_path: str = MODEL_PATH):
    """Visualiza los 32 kernels aprendidos de conv1."""
    from models import ChannelFusionCNN

    model = ChannelFusionCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))

    kernels = model.conv1.weight.data.numpy()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(kernels[i, 0], cmap='RdBu_r')
        ax.set_title(f'K{i}', fontsize=8)
        ax.axis('off')

    plt.suptitle('32 Kernels (3×5) aprendidos - Conv1', fontweight='bold')
    plt.tight_layout()
    plt.savefig('kernels_entrenados.png', dpi=150)
    plt.show()


def visualizar_activaciones(model, signals, labels, idx):
    """Muestra feature maps de conv1 para una ventana."""
    model.eval()
    x = torch.from_numpy(signals[idx]).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        act = model.conv1(x).squeeze().numpy()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(act[i], cmap='viridis', aspect='auto')
        ax.set_title(f'F{i}', fontsize=8)
        ax.axis('off')

    label_str = 'CRISIS' if labels[idx] == 1 else 'NORMAL'
    plt.suptitle(f'Activaciones Conv1 - {label_str} (idx={idx})', fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
def visualizar_pipeline_cnn():
    """
    Genera un diagrama visual del pipeline completo del modelo ChannelFusionCNN.
    Muestra el flujo desde los datos EEG hasta la clasificación.
    """
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Colores
    c_input = '#3498db'
    c_conv = '#e74c3c'
    c_pool = '#f39c12'
    c_fc = '#9b59b6'
    c_output = '#27ae60'
    c_arrow = '#2c3e50'

    def draw_box(x, y, w, h, color, text, fontsize=9):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                      facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=c_arrow, lw=2))

    # === TÍTULO ===
    ax.text(9, 13.5, 'Pipeline ChannelFusionCNN - Detección de Crisis Epilépticas',
            ha='center', fontsize=16, fontweight='bold')

    # === INPUT ===
    draw_box(0.5, 11, 3.5, 1.5, c_input, 'INPUT\nEEG Raw\n(batch, 1, 21, 128)')

    # Dibujar representación visual del input
    ax.add_patch(patches.Rectangle((4.5, 11.2), 1.2, 1.1, facecolor='lightblue', edgecolor='black'))
    ax.text(5.1, 11.75, '21 ch\n×\n128 t', ha='center', va='center', fontsize=7)

    # === BLOQUE 1 ===
    draw_arrow(4, 11.75, 6.5, 11.75)

    draw_box(6.5, 11, 3, 1.5, c_conv, 'CONV1\nkernel(3,5)\n32 filtros')
    draw_arrow(9.5, 11.75, 10.5, 11.75)

    draw_box(10.5, 11, 2.5, 1.5, c_pool, 'BatchNorm\n+ ReLU\n+ Pool(2,2)')

    ax.text(14, 11.75, '→ (b, 32, 10, 64)', fontsize=9, va='center')

    # === BLOQUE 2 ===
    draw_arrow(13, 11, 13, 9.5)

    draw_box(6.5, 8, 3, 1.5, c_conv, 'CONV2\nkernel(3,5)\n64 filtros')
    draw_arrow(9.5, 8.75, 10.5, 8.75)

    draw_box(10.5, 8, 2.5, 1.5, c_pool, 'BatchNorm\n+ ReLU\n+ Pool(2,2)')

    ax.text(14, 8.75, '→ (b, 64, 5, 32)', fontsize=9, va='center')
    draw_arrow(13, 8, 13, 6.5)

    # === BLOQUE 3 ===
    draw_box(6.5, 5, 3, 1.5, c_conv, 'CONV3\nkernel(3,3)\n128 filtros')
    draw_arrow(9.5, 5.75, 10.5, 5.75)

    draw_box(10.5, 5, 2.5, 1.5, c_pool, 'BatchNorm\n+ ReLU\n+ Pool(2,2)')

    ax.text(14, 5.75, '→ (b, 128, 2, 16)', fontsize=9, va='center')

    # === FLATTEN ===
    draw_arrow(11.75, 5, 11.75, 3.8)
    draw_box(10, 2.8, 3.5, 1, '#bdc3c7', 'FLATTEN\n(b, 4096)')

    # === FC LAYERS ===
    draw_arrow(11.75, 2.8, 11.75, 1.8)
    draw_box(6.5, 0.8, 3, 1, c_fc, 'FC1\n4096 → 256\n+ ReLU + Dropout')
    draw_arrow(9.5, 1.3, 10.5, 1.3)

    draw_box(10.5, 0.8, 2.5, 1, c_fc, 'FC2\n256 → 2')
    draw_arrow(13, 1.3, 14.5, 1.3)

    # === OUTPUT ===
    draw_box(14.5, 0.5, 3, 1.5, c_output, 'OUTPUT\nSoftmax\n[P(normal), P(crisis)]')

    # === LEYENDA DE KERNEL ===
    ax.add_patch(patches.Rectangle((0.5, 5), 4.5, 3.5, facecolor='lightyellow',
                                   edgecolor='black', linestyle='--'))
    ax.text(2.75, 8.2, 'Kernel Conv1 (3×5)', ha='center', fontsize=10, fontweight='bold')

    # Mini visualización del kernel
    for i in range(3):
        for j in range(5):
            ax.add_patch(patches.Rectangle((1 + j * 0.6, 6.8 - i * 0.5), 0.5, 0.4,
                                           facecolor='coral', edgecolor='black'))
    ax.text(2.75, 5.3, '3 canales (espacial)\n5 muestras (temporal)',
            ha='center', fontsize=8)

    # === LEYENDA DE COLORES ===
    legend_y = 3.5
    legend_items = [
        (c_input, 'Input'),
        (c_conv, 'Convolución'),
        (c_pool, 'Normalización + Pooling'),
        (c_fc, 'Fully Connected'),
        (c_output, 'Output')
    ]

    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(patches.Rectangle((0.5, legend_y - i * 0.5), 0.4, 0.35,
                                       facecolor=color, edgecolor='black'))
        ax.text(1.1, legend_y - i * 0.5 + 0.17, label, fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig('pipeline_cnn_completo.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()


if __name__ == "__main__":
    signals, labels = cargar_datos("chb01")

    # Cargar datos
    duracion = calcular_duracion_paciente(signals, fs=128)
    signals, labels = cargar_datos("chb01")

    # Visualización principal: qué captura el kernel
    visualizar_kernel_captura(signals, labels)

    # Otras visualizaciones
    plot_channel_fusion_input(signals, labels)
    comparar_estadisticas(signals, labels)
    # Pipeline visual
    visualizar_pipeline_cnn()
    # Kernels del modelo (si existe)
    try:
        visualizar_kernels_modelo()
    except FileNotFoundError:
        print("Modelo no encontrado, saltando visualización de kernels")
