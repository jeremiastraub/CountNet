"""Run script for CountNet training and validation"""

from typing import Union

from torch.utils.data import DataLoader

from CountNet.utils import (load_yml, initialize_trainer,
                            parse_training_kwargs, parse_validation_kwargs)

# -----------------------------------------------------------------------------

# Get the configurations
datasets_cfg = load_yml("CountNet/data/datasets_cfg.yml")
run_cfg = load_yml("CountNet/run_cfg.yml")

model_cfg = run_cfg['CountNet']
trainer_cfg = run_cfg['Trainer']
run_training_cfg = run_cfg.get('training', None)
run_validation_cfg = run_cfg.get('validation', None)


if __name__ == '__main__':
    trainer = initialize_trainer(trainer_cfg, model_cfg=model_cfg,
                                              dset_cfg=datasets_cfg)
    
    # TODO Allow loading a previously saved checkpoint
    
    if run_training_cfg is not None:
        print(f"Starting training...\n\nModel configuration:\n{model_cfg}\n\n"
              f"Training configuration:\n{run_training_cfg}\n")
        losses, validations = trainer.train_model(
                                **parse_training_kwargs(run_training_cfg))

        # Plot losses over training iteration
        import matplotlib.pyplot as plt
        if losses:
            plt.plot(losses)
            plt.title('Loss')
            plt.show() # opens separate window, pauses until it is closed
            # plt.savefig("output.png")

    if run_validation_cfg is not None:
        print("Starting validation...\n\nValidation configuration:\n"
              f"{run_validation_cfg}\n")
        scores = trainer.validate_model(
                                **parse_validation_kwargs(run_validation_cfg))

        # Print final scores in terminal
        for m, s in zip(run_validation_cfg['metrics'], scores):
            print(f"{m}: {s}")
