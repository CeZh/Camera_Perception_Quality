# Camera_Perception_Quality

# Introduction
This is the official implementation of the paper [A Quality Index Metric and Method for Online Self-Assessment of Autonomous Vehicles Sensory Perception](https://arxiv.org/abs/2203.02588). 

The proposed model is superpixel attention-based neural network, which combines superpixel and pixel features for camera-based perception quality index regression. Current model support [BDD100k](https://www.bdd100k.com/), [NuScene](https://www.nuscenes.org/), and [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset. Detailed implementation and labeling can be seen under the Implementation section

![Network Architecture](/project_images/Network_Architecture.png)
SPA-NET Model Architecture
![Project Workflow](/project_images/Project_Flowchart.png)
Camera Perception Quality Index Project Flowchart

### Citation
```
@inproceedings{Zhang2022AQI,
  title={A Quality Index Metric and Method for Online Self-Assessment of Autonomous Vehicles Sensory Perception},
  author={Ce Zhang and Azim Eskandarian},
  year={2022}
}
```

# Installation
The code was tested on Ubuntu 20.04 with [Anaconda](https://www.anaconda.com/) Python 3.8, CUDA 11.3, and PyTorch v1.0. It should be compatible with PyTorch > 1.7 & Python 3.7. (If 30x GPUs, require CUDA 11.x. If 20X GPUs, CUDA 10.x should also work for this project)

## Install Anaconda Environment
```
conda create --name Camera_Perception_Quality python=3.8
conda activate Camera_Perception_Quality
```
## Install Pytorch and Other Python Libararies
### Pytorch
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
### Required Libraries
```
pip install -r requirements.txt
```

# Implementation
The implementation of this project contains a demo module and a training module
## Demo
The pre-trained models are available [here](https://drive.google.com/drive/folders/11i3vIhq1Xhe2tIgEBrQTtan49h6-Kj4g?usp=sharing) (Google Drive). Run ``` demp.py ``` with the selection of the desired dataset.
### Example (BDD100k Dataset)

1. Go to the BDD dataset's configuration file under ```configs/bdd100k/super_vit_linear.yaml```
2. Change the ```train_image_path, train_label_path, val_image_path, val_label_path``` to the desired directory
3. run ```demo.py --configs ./configs/bdd100k/super_vit_linear.yaml --file_dir ./demo_image/img1.jpg```

## Training
The model can be trained under BDD100K, NuScene, and KITTI datasets. The corresponding labels for each datasets are [here](https://drive.google.com/drive/folders/13WnUMU37wEerasEczFGrfrbtDqjPDyaS?usp=sharing)

### Example
1. Go to the BDD dataset's configuration file under ```configs/bdd100k/super_vit_linear.yaml```
2. Change the ```train_image_path, train_label_path, val_image_path, val_label_path``` to the desired directory
3. run ```train.py --configs ./configs/bdd100k/super_vit_linear.yaml --file_dir ./demo_image/img1.jpg```


## License
[MIT](https://choosealicense.com/licenses/mit/)
