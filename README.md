# ultralytics-YOLOv3-Cluster-NMS
Cluster-NMS into YOLOv3 Pytorch
This is the code for our papers:
 - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
 - [Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/1904.02689)

```
@inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}
```
# Introduction

This directory contains PyTorch YOLOv3 software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information please visit https://www.ultralytics.com.

# Description

The https://github.com/ultralytics/yolov3 repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.

# Requirements

Python 3.7 or later with all `pip install -U -r requirements.txt` packages including `torch >= 1.5`. Docker images come with all dependencies preinstalled. Docker requirements are: 
- Nvidia Driver >= 440.44
- Docker Engine - CE >= 19.03

# mAP

<i></i>                      |Size |COCO mAP<br>@0.5...0.95 |COCO mAP<br>@0.5 
---                          | ---         | ---         | ---
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |320 |14.0<br>28.7<br>30.5<br>**37.7** |29.1<br>51.8<br>52.3<br>**56.8**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |416 |16.0<br>31.2<br>33.9<br>**41.2** |33.0<br>55.4<br>56.9<br>**60.6**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |512 |16.6<br>32.7<br>35.6<br>**42.6** |34.9<br>57.7<br>59.5<br>**62.4**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |608 |16.6<br>33.1<br>37.0<br>**43.1** |35.4<br>58.2<br>60.7<br>**62.8**

- mAP@0.5 run at `--iou-thr 0.5`, mAP@0.5...0.95 run at `--iou-thr 0.7`
- Darknet results: https://arxiv.org/abs/1804.02767

## Cluster-NMS
This repo only focuses on NMS improvement.

#### Hardware
 - 2 GTX 1080 Ti
 - Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz
 
 | Image Size | Model  | NMS  | FPS  | box AP | box AP75 | box AR100 |
|:----:|:-------------:|:------------------------------------:|:----:|:----:|:----:|:----:|
| 608  | YOLOv3-SPP-ultralytics |                 Fast NMS               | 85.5 | 42.2 | 45.1 | 60.1 |
| 608  | YOLOv3-SPP-ultralytics |               Original NMS             | 14.6 | 42.6 | 45.8 | 62.5 | 
| 608  | YOLOv3-SPP-ultralytics |        Original NMS Torchvision        | 95.2 | 42.6 | 45.8 | 62.5 | 
| 608  | YOLOv3-SPP-ultralytics |               Cluster-NMS              | 82.6 | 42.6 | 45.8 | 62.5 | 
| 608  | YOLOv3-SPP-ultralytics |             Cluster-DIoU-NMS           | 76.9 | 42.7 | 46.2 | 63.4 | 
| 608  | YOLOv3-SPP-ultralytics |               Weighted-NMS             | 11.2 | 42.9 | 46.4 | 62.7 |
| 608  | YOLOv3-SPP-ultralytics |          Weighted Cluster-NMS          | 68.0 | 42.9 | 46.4 | 62.7 |
| 608  | YOLOv3-SPP-ultralytics |       Weighted + Cluster-DIoU-NMS      | 64.9 | 43.1 | 46.8 | 63.7 |
| 608  | YOLOv3-SPP-ultralytics |       Weighted + Torchvision NMS       | 88.5 | 42.8 | 46.3 | 63.0 |
## Discussion

Note that Torchvision NMS has the fastest speed, that is owing to CUDA imprementation and engineering accelerations (like upper triangular IoU matrix only). However, our Cluster-NMS requires less iterations for NMS and can also be further accelerated by adopting engineering tricks. Almost completed at the same time as the work of our paper is Glenn Jocher's Torchvision NMS + merge. First, we do Torchvision NMS, then convert the output to vector to multiply the IoU matrix. Also, for merge/Weighted NMS, the IoU matrix is no need to be square shape \[n\times n\]. It can be \[m\times n\] to save more time, where \[m\] is the boxes that NMS outputs.

# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)
