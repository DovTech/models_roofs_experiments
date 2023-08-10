# Application of models in roof segmentation
This repository contains UNet and YOLO models configured for better segmentation and detection of buildings, respectively. 
The repository also contains commits of late changes in the code of both training and testing pipelines.

## Dataset
Dataset was taken from AIcrowd Mapping Challenge. The link is left below

## Tasks
The main task is the segmentation of the roofs of buildings in the images.
Initially, the tasks included segmentation and detection of roofs of buildings in images, but at the moment the detection task 
has been removed due to the curve of the construction of bounding boxes by dataset. This problem has also been encountered by 
other users of this dataset.

# Model pipelines
## UNet
UNet has been modified to work with RGB images and to create building masks more accurately.
![image](https://github.com/DovTech/unet_roofs_experiments/assets/90236671/edbcd50b-12da-40a2-b476-df2460573543)

## YOLOv1
YOLOv1 represents my implementation of the YOLOv1 paper architecture.<br>
At the moment, the file contains my implementation of the YOLOv1 architecture according to the article of its creators without the rest of the pipeline.
![image](https://github.com/DovTech/unet_roofs_experiments/assets/90236671/a7169e62-9581-48a7-98fc-634b8046b9e2)

# References 
1. U-Net: Convolutional Networks for Biomedical
Image Segmentation [https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1505.04597.pdf)
2. You Only Look Once:
Unified, Real-Time Object Detection [https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1506.02640.pdf)
3. AIcrowd Mapping Challenge:
https://www.aicrowd.com/challenges/mapping-challenge
