#!/bin/bash

#SBATCH --job-name=lightning_test # Nome del job
#SBATCH --output=slurm_output_%j.out # File di output
#SBATCH --ntasks=1                # Esegui un singolo task
#SBATCH --cpus-per-task=4         # Richiedi 4 core CPU
#SBATCH --mem=8G                  # Richiedi 8GB di memoria
#SBATCH --time=00:15:00           # Tempo massimo di esecuzione
#SBATCH --gres=gpu:1              # Richiedi 1 GPU generica

# Stampa informazioni utili sull'ambiente
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# Carica i moduli necessari (adatta al tuo cluster)
# module load cuda/11.8
# module load anaconda/2023.03

# Crea ambiente virtuale e installa dipendenze
#python -m venv myenv_lightning
#source myenv_lightning/bin/activate
#source /home1/vvitale/workspace/slurm_toy/toy_venv/bin/activate
#pip install --upgrade pip
#pip install -r requirements.txt

echo "Avvio dello script di training PyTorch Lightning..."

srun python /home1/vvitale/workspace/slurm_toy/train.py

echo "Job terminato."