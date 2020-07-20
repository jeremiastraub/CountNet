"""Run script for CountNet training and validation"""

from typing import Union

from torch.utils.data import DataLoader

from ..data import CrowdCountingDataset, transforms
from ..network import CountNet, Trainer
from utils import load_yml, initialize_trainer

# -----------------------------------------------------------------------------

# Get the configurations
datasets_cfg = load_yml("data/datasets_cfg.yml")
run_cfg = load_yml("network/run_cfg.yml")

model_cfg = run_cfg['CountNet']
trainer_cfg = run_cfg['Trainer']
run_training_cfg = run_cfg.get('training', None)
run_validation_cfg = run_cfg.get('validation', None)


if __name__ == '__main__':
    trainer = initialize_trainer(trainer_cfg, dset_cfg=datasets_cfg)
    
    # TODO Allow loading a previously saved checkpoint
    
    if run_training_cfg is not None:
        print(f"Starting training...\n\nModel configuration:\n{model_cfg}\n\n"
              f"Training configuration:\n{run_training_cfg}")
        trainer.train_model(**run_training_cfg)

    if run_validation_cfg is not None:
        print("Starting validation...\n\nValidation configuration:\n"
              f"{run_validation_cfg}")
        trainer.validate_model(**run_validation_cfg)
