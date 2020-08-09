"""Run script for CountNet training (and subsequent validation)"""

from CountNet.utils import (load_yml, initialize_trainer,
                            parse_training_kwargs, parse_validation_kwargs)

# -----------------------------------------------------------------------------
# Change these paths to load the configurations from different files
DATASET_CFG_PATH = "CountNet/data/datasets_cfg.yml"
RUN_CFG_PATH = "CountNet/run_cfg.yml"

# -----------------------------------------------------------------------------
# Get the configurations
datasets_cfg = load_yml(DATASET_CFG_PATH)
run_cfg = load_yml(RUN_CFG_PATH)

model_cfg = run_cfg['CountNet']
trainer_cfg = run_cfg['Trainer']
run_training_cfg = run_cfg.get('training', None)
run_validation_cfg = run_cfg.get('validation', None)

if __name__ == '__main__':
    trainer = initialize_trainer(trainer_cfg, model_cfg=model_cfg,
                                              dset_cfg=datasets_cfg)
    
    if run_training_cfg is not None:
        print(f"Starting training...\n\nModel configuration:\n{model_cfg}\n\n"
              f"Training configuration:\n{run_training_cfg}\n")
        _ = trainer.train_model(**parse_training_kwargs(run_training_cfg))

    if run_validation_cfg is not None:
        print("Starting validation...\n\nValidation configuration:\n"
              f"{run_validation_cfg}\n")
        scores = trainer.validate_model(
                                **parse_validation_kwargs(run_validation_cfg))

        # Print final scores in terminal
        for m, s in scores.items():
            print(f"{m}: {s}")
