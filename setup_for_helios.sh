salloc --partition=plgrid-gpu-gh200 --gres=gpu:1 --time=4:00:00 --mem=64G
srun --pty bash

ml load ML-bundle/25.10
python -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt

export PYTHONPATH=/net/scratch/hscra/plgrid/${USER:3}/zml:$PYTHONPATH
python scripts/unlearn.py --config experiments/exp016_unhype_fire/config.yaml 