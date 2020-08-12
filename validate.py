"""Run script for CountNet validation"""

from CountNet.utils import (load_yml, initialize_trainer,
                            parse_validation_kwargs)

# -----------------------------------------------------------------------------
# Change these paths to load the configurations from different files
DATASET_CFG_PATH = "CountNet/data/datasets_cfg.yml"
RUN_CFG_PATH = "CountNet/run_cfg.yml"

# -----------------------------------------------------------------------------
# Get the configurations
datasets_cfg = load_yml(DATASET_CFG_PATH)
run_cfg = load_yml(RUN_CFG_PATH)
validation_cfg = run_cfg.get('validation', None)

assert validation_cfg is not None, "No validation configuration found!"

model_cfg = run_cfg['CountNet']
trainer_cfg = run_cfg['Trainer']

# Set 'loader_train' entry to 'None' such that the training data is not loaded
trainer_cfg['loader_train'] = None

if not 'validate_run' in trainer_cfg:
    raise ValueError("No tag found at 'Trainer.validate_run'! The tag "
                     "specifies the run to be validated.")

if __name__ == '__main__':
    trainer = initialize_trainer(trainer_cfg, model_cfg=model_cfg,
                                              dset_cfg=datasets_cfg)

    print("Starting validation...\n\nValidation configuration:\n"
          f"{validation_cfg}\n")
    scores = trainer.validate_model(**parse_validation_kwargs(validation_cfg))
