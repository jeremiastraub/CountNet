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

    scores = trainer.validate_model(**parse_validation_kwargs(validation_cfg))

    # FIXME This is only temporary!

    # Visualize a few predictions
    img_batch, dm_batch = next(iter(trainer.loader_test))

    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    print("One example batch:...")
    trainer.model.to(device='cpu')
    for i, _ in zip(range(6), range(img_batch.shape[0])):
        img = img_batch[i,...]
        dm = dm_batch[i,...]

        plt.imshow(img.permute(1,2,0).numpy().squeeze())
        plt.imshow(dm.permute(1,2,0).numpy().squeeze(), alpha=0.8)
        plt.savefig(f"{i}_output_true.png", bbox_inches='tight')
        plt.close()

        res = trainer.model(img.unsqueeze(0)).detach()

        mse = nn.MSELoss(reduction='sum')
        print("MSE: ", mse(res.squeeze(), dm.squeeze()).item())

        res = res.numpy().squeeze()
        plt.imshow(res, vmin=0)
        plt.savefig(f"{i}_output_res.png", bbox_inches='tight')
        plt.close()

        print("True sum: ", np.sum(dm.permute(1,2,0).numpy().squeeze()))
        print("Prediction: ", np.sum(res))
