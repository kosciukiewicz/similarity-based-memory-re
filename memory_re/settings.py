import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent.resolve()

load_dotenv(verbose=True)

_PRETRAINED_MODELS_DIR = os.environ.get('PRETRAINED_MODELS_DIR')
PRETRAINED_MODELS_DIR = Path(_PRETRAINED_MODELS_DIR) if _PRETRAINED_MODELS_DIR else None
WANDB_PROJECT_NAME = os.environ.get('WANDB_PROJECT_NAME')

STORAGE_DIR = PROJECT_DIR / 'storage'

CONFIGS_DIR = PROJECT_DIR / 'configs'
DATA_DIR = STORAGE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
DATASETS_DIR = DATA_DIR / 'datasets'
PREDICTIONS_DIR = STORAGE_DIR / 'predictions'
SAVED_MODELS_DIR = STORAGE_DIR / 'saved_models'
CHECKPOINTS_DIR = STORAGE_DIR / 'checkpoints'
EXPERIMENTS_DIR = STORAGE_DIR / 'experiments'
WANDB_RUNS_DIR = EXPERIMENTS_DIR / 'wandb'
