#!/bin/bash
#SBATCH -J ChannelFusion                    # Nombre del trabajo
#SBATCH -n 1                                # 1 tarea
#SBATCH -N 1                                # 1 nodo
#SBATCH -D /fhome/maed01/-Epileptic-seizures-in-ElectroEncephalogram/ChannelFusion_Sistema1/ # Directorio de trabajo correcto
#SBATCH -t 4-00:05                          # Tiempo máximo
#SBATCH -p tfg                              # Partición
#SBATCH --mem 40G                           # Memoria RAM
#SBATCH -o logs/%x_%j.out                   # Archivo de salida entardar (nombre_jobid.out)
#SBATCH -e logs/%x_%j.err                   # Archivo de errores (nombre_jobid.err)
#SBATCH --gres gpu:1                        # Solicitar 1 GPU
#SBATCH --cpus-per-task=4                   # CPUs por tarea

# Crear directorio de logs si no existe para evitar errores de escritura de SBATCH -o/-e (puede requerir existir antes de sbatch)
mkdir -p logs 

# Activar entorno virtual
source /fhome/maed01/-Epileptic-seizures-in-ElectroEncephalogram/MyVirtualEnv/bin/activate

# Ejecutar script
# Al usar -D arriba, ya estamos en la carpeta correcta, así que podemos llamar a python directamente sobre el archivo local
srun python minimain.py