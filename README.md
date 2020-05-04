# Ultralytics-YOLOv3-Cluster-NMS
## Cluster-NMS into YOLOv3 Pytorch

This is the code for our paper:
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

This repo only focuses on NMS improvement based on https://https://github.com/ultralytics/yolov3.

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

#### Hardware
 - 2 GTX 1080 Ti
 - Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz
 
Evaluation command: `python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt`

AP reports on `coco 2014 minival`.

 | Image Size | Model  | NMS  | FPS  | box AP | box AP75 | box AR100 |
|:----:|:-------------:|:------------------------------------:|:----:|:----:|:----:|:----:|
| 608  | YOLOv3-SPP-ultralytics |                 Fast NMS               | 85.5     | 42.2     | 45.1     | 60.1     |
| 608  | YOLOv3-SPP-ultralytics |               Original NMS             | 14.6     | 42.6     | 45.8     | 62.5     | 
| 608  | YOLOv3-SPP-ultralytics |                 DIoU-NMS               | 7.9      | 42.7     | 46.2     | 63.4     | 
| 608  | YOLOv3-SPP-ultralytics |        Original NMS Torchvision        | **95.2** | 42.6     | 45.8     | 62.5     | 
| 608  | YOLOv3-SPP-ultralytics |               Cluster-NMS              | 82.6     | 42.6     | 45.8     | 62.5     | 
| 608  | YOLOv3-SPP-ultralytics |             Cluster-DIoU-NMS           | 76.9     | 42.7     | 46.2     | 63.4     | 
| 608  | YOLOv3-SPP-ultralytics |               Weighted-NMS             | 11.2     | 42.9     | 46.4     | 62.7     |
| 608  | YOLOv3-SPP-ultralytics |          Weighted Cluster-NMS          | 68.0     | 42.9     | 46.4     | 62.7     |
| 608  | YOLOv3-SPP-ultralytics |       Weighted + Cluster-DIoU-NMS      | 64.9     | **43.1** | **46.8** | **63.7** |
| 608  | YOLOv3-SPP-ultralytics |         Merge + Torchvision NMS        | 88.5     | 42.8     | 46.3     | 63.0     |
| 608  | YOLOv3-SPP-ultralytics |      Merge + DIoU + Torchvision NMS    | 82.5     | 43.0     | 46.6     | 63.2     |
## Discussion

 - Merge NMS is a simplified version of Weighted-NMS. It just use score vector for weighted coordinates, not combine score and IoU.
 
 - We further incorporate DIoU into NMS for YOLOv3 which can get higher AP and AR.
 
 - Note that Torchvision NMS has the fastest speed, that is owing to CUDA imprementation and engineering accelerations (like upper triangular IoU matrix only). However, our Cluster-NMS requires less iterations for NMS and can also be further accelerated by adopting engineering tricks. Almost completed at the same time as the work of our paper is Glenn Jocher's Torchvision NMS + Merge. First, we do Torchvision NMS, then convert the output to vector to multiply the IoU matrix. Also, for Merge NMS, the IoU matrix is no need to be square shape `n*n`. It can be `m*n` to save more time, where `m` is the boxes that NMS outputs.
 
 - Currently, Torchvision NMS use IoU as criterion, not DIoU. However, if we directly replace IoU with DIoU in Original NMS, it will costs much more time due to the sequence operation. Now, Cluster-DIoU-NMS will significantly speed up DIoU-NMS and obtain exactly the same result.
 
 - Torchvision NMS is a function in Torchvision>=0.4, and our Cluster-NMS can be applied to any projects that use low version of Torchvision and other deep learning frameworks as long as it can do matrix operations. No other import, no need to compile, less iteration, fully GPU-accelerated and better performance.
# Citation

[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)
