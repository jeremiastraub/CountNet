# CountNet
Using supervised deep learning to count people in images.

### Datasets
- **UCSD**: consists of 2000 pedestrian frames of size 238x158 from a video sequence along with ground truth annotations of each pedestrian in every fifth frame. The dataset contains a total of 49885 pedestrian instances and it is split into training and test set
- **Mall**: has diverse illumination conditions and crowd densities. Moreover, the scene contained in the dataset has severe perspective distortion. The dataset also presents severe occlusions. The video sequence in the dataset consists of 2000 frames of size 320x240 with 6000 instances of labeled pedestrians split into training and evaluation set
- **UCF_CC_50**: includes a wide range of densities and diverse scenes with varying perspective distortion. It contains a total of 50 images with an average of 1280 individuals per image. Due to the limited number of images is recommended to define a cross-validation protocol for training and testing.
- **WorldExpo ’10**: consists of a total of 3980 frames of size 576x720 with 199923 labeled pedestrians. It is split into training and evaluation set
- **Shanghai Tech**: consists of 1198 images with 330165 annotated heads. The dataset successfully attempts to create a challenging dataset with diverse scene types and varying density levels.
- [**JHU-CROWD++**](http://www.crowd-counting.com/): A comprehensive dataset with 4,372 images and 1.51 million annotations for a variety of diverse scenarios and environmental conditions.

More recent datasets are: Smartcity, UCF-QNRF, City Street, FDST, Crowd Surveillance, JHU-CROWD,
DLR-ACD, DroneCrowd, GCC, NWPU-Crowd

### Papers & Articles
- Objects Counting by Estimating a Density Map With Convolutional Neural Networks: https://towardsdatascience.com/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks-c01086f3b3ec
- Introduction to Crowd counting: https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
- CSRNet - Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes: https://arxiv.org/pdf/1802.10062.pdf
- Deep Spatial Regression Model for Image Crowd Counting: https://arxiv.org/pdf/1710.09757.pdf
- Iterative Crowd Counting: https://www3.cs.stonybrook.edu/~minhhoai/papers/crowdcountingECCV18.pdf
