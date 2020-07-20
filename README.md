# CountNet
Using supervised deep learning to count people in images.

### Datasets
- **Mall**: has diverse illumination conditions and crowd densities. Moreover, the scene contained in the dataset has severe perspective distortion. The dataset also presents severe occlusions. The video sequence in the dataset consists of 2000 frames of size 640x480 with 6000 instances of labeled pedestrians split into training and evaluation set
- **UCF_CC_50**: includes a wide range of densities and diverse scenes with varying perspective distortion. It contains a total of 50 images with an average of 1280 individuals per image. Due to the limited number of images is recommended to define a cross-validation protocol for training and testing.
- **Shanghai Tech**: consists of 1198 images with 330165 annotated heads. The dataset successfully attempts to create a challenging dataset with diverse scene types and varying density levels.
- **UCF-QNRF**: contains 1535 images which are divided into train and test sets of 1201 and 334 images respectively. It is very diverse in terms of prepectivity, image resolution, crowd density and the scenarios which a crowd exist. Is most suitable for training very deep Convolutional Neural Networks (CNNs) since it contains order of magnitude more annotated humans in dense crowd scenes than any other available crowd counting datase.
- **UCSD**: consists of 2000 pedestrian frames of size 238x158 from a video sequence along with ground truth annotations of each pedestrian in every fifth frame. The dataset contains a total of 49885 pedestrian instances and it is split into training and test set
- **WorldExpo ’10**: consists of a total of 3980 frames of size 576x720 with 199923 labeled pedestrians. It is split into training and evaluation set
- [**JHU-CROWD++**](http://www.crowd-counting.com/): A comprehensive dataset with 4,372 images and 1.51 million annotations for a variety of diverse scenarios and environmental conditions.

More recent datasets are: Smartcity, UCF-QNRF, City Street, FDST, Crowd Surveillance, JHU-CROWD, DLR-ACD, DroneCrowd, GCC, NWPU-Crowd

### Papers & Articles
Survey:
- Introduction to Crowd counting: https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
- Objects Counting by Estimating a Density Map With Convolutional Neural Networks: https://towardsdatascience.com/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks-c01086f3b3ec
- CNN-based Density Estimation and Crowd Counting - A Survey: https://arxiv.org/pdf/2003.12783v1.pdf
- A Survey of Recent Advances in CNN-based Single Image Crowd Counting and Density Estimation: http://arxiv.org/abs/1707.01202
- Convolutional neural networks for crowd behaviour analysis: a survey: http://link.springer.com/10.1007/s00371-018-1499-5

Other:
- CSRNet - Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes: https://arxiv.org/pdf/1802.10062.pdf
- W-Net: Reinforced U-Net for Density Map Estimation: http://arxiv.org/abs/1903.11249
- Crowd Counting and Density Estimation by Trellis Encoder-Decoder Networks: https://ieeexplore.ieee.org/document/8954254/
- Deep Spatial Regression Model for Image Crowd Counting: https://arxiv.org/pdf/1710.09757.pdf
- Iterative Crowd Counting: https://www3.cs.stonybrook.edu/~minhhoai/papers/crowdcountingECCV18.pdf
- Microscopy Cell Counting with Fully Convolutional Regression Networks: https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf
- People Counting in Dense Crowd Images Using Sparse Head Detections: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8283650
