# CountNet
Using supervised deep learning to count people in images. The CountNet model is based on the U-Net architecture with multiscale inception blocks as decoder. For more details, have a look at the [Project Report](report.pdf).


## Datasets
This repository includes four ready-to-use datasets. Any new dataset requires its own derived dataset class (see [datasets.py](CountNet/data/datasets.py)). So far, the CountNet model could only be trained successfully on the *Mall* dataset.

- **Mall**: has diverse illumination conditions and crowd densities. Moreover, the scene contained in the dataset has severe perspective distortion. The dataset also presents severe occlusions. The video sequence in the dataset consists of 2000 frames of size 640x480 with 6000 instances of labeled pedestrians split into training (1600) and evaluation set (400).
- **UCF_CC_50**: includes a wide range of densities and diverse scenes with varying perspective distortion. It contains a total of 50 images with an average of 1280 individuals per image. Due to the limited number of images is recommended to define a cross-validation protocol for training and testing. However, it is currently split into training (40) and test (10) set.
- **Shanghai Tech** (2 parts: A and B): consists of 1198 images with 330165 annotated heads. The dataset successfully attempts to create a challenging dataset with diverse scene types and varying density levels.


## Generating ground-truth
Before training the model the first time, the ground-truth density maps need to be created. These are not included in this repository as they take up disk space of around 1GB (when creating them for all four datasets). We provide a [script](CountNet/datasets/generate_density_maps.py) for automatic ground-truth generation. Run it via:

```bash
cd CountNet/datasets
python generate_density_maps.py
```
(This might take a while.)


## Configuration
We aimed at designing the project such that the pipeline of configuring the datasets and training, starting a training run, and evaluating previous runs is easy to use and that results are reproducible. For example, we use the YAML language for most configurations, which makes it easy to adjust parameters and configurations are stored alongside with model output.

The preprocessing transformations applied to the image data (e.g., downscaling, random crops) can be configured in [the dataset configuration](CountNet/data/datasets_cfg.yml)

The core of the training and validation configuration is the [run configuration](CountNet/run_cfg.yml).


## Training the model
Start a training run via:

```bash
python run.py
```
This automatically reads in all current dataset and training configurations and stores a checkpoint as well as the configuration in an `output` folder.

Note that previous checkpoints can be loaded via `Trainer.load_from` in the [run configuration](CountNet/run_cfg.yml). That way, training is continued from that checkpoint.


## Validating the model
To validate a model trained previously, specify the model checkpoint to be loaded as the `Trainer.validate_run` entry in the [run configuration](CountNet/run_cfg.yml). Then run the validation script:

```bash
python validate.py
```
We also provide various evaluation and plotting utilities in the [plotting module](CountNet/plotting).


## Related Papers & Articles
Surveys on crowd counting:
- Introduction to Crowd counting: https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
- Objects Counting by Estimating a Density Map With Convolutional Neural Networks: https://towardsdatascience.com/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks-c01086f3b3ec
- CNN-based Density Estimation and Crowd Counting - A Survey: https://arxiv.org/pdf/2003.12783v1.pdf
- A Survey of Recent Advances in CNN-based Single Image Crowd Counting and Density Estimation: http://arxiv.org/abs/1707.01202
- Convolutional neural networks for crowd behaviour analysis: a survey: http://link.springer.com/10.1007/s00371-018-1499-5

Models:
- CSRNet - Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes: https://arxiv.org/pdf/1802.10062.pdf
- W-Net: Reinforced U-Net for Density Map Estimation: http://arxiv.org/abs/1903.11249
- Crowd Counting and Density Estimation by Trellis Encoder-Decoder Networks: https://ieeexplore.ieee.org/document/8954254/
- Deep Spatial Regression Model for Image Crowd Counting: https://arxiv.org/pdf/1710.09757.pdf
- Iterative Crowd Counting: https://www3.cs.stonybrook.edu/~minhhoai/papers/crowdcountingECCV18.pdf
- Microscopy Cell Counting with Fully Convolutional Regression Networks: https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf
- People Counting in Dense Crowd Images Using Sparse Head Detections: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8283650
