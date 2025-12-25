# GF-YOLO: Enhanced Small Object Detection for UAV Aerial Imagery

## 📝 Overview

This repository contains the implementation of **GF-YOLO**, an improved YOLO model based on YOLOv8n specifically designed for small object detection in UAV aerial images. Our method addresses the challenges of high proportion of small targets, substantial scale variations, and complex backgrounds while maintaining computational efficiency suitable for edge devices.

## 🚀 Key Improvements

### 1. Scale-Adaptive Network Architecture

- **P2 Layer Addition**: Introduced a dedicated P2 detection layer for enhanced small-object detection capability
- **P5 Layer Removal**: Removed the P5 layer designed for large objects to reduce computational overhead
- **Shallow Channel Expansion (SCE)**: Proposed strategy to increase channel dimensions of shallow backbone layers, capturing more comprehensive features for small objects

### 2. Global Feature Fusion Architecture (GFF)

- **Multi-scale Feature Fusion (MFF)**: Efficient cross-scale semantic information propagation
- **Weighted Feature Fusion (WFF)**: Deep feature integration through adaptive weighting
- **Cascaded Strategy**: Combines MFF and WFF modules for optimal feature fusion in the neck network

### 3. Dynamic Detection Head (DyHead)

- **Multiple Attention Mechanisms**: Spatial, channel, and scale attention integration
- **Adaptive Response Adjustment**: Dynamic weight adjustment across different levels, spatial locations, and channels
- **Enhanced Feature Representation**: Improved localization accuracy for small objects

## 🏗️ Model Architecture

```

```

<img src="Experimental Data/Model Architecture.png" alt="图片2" style="zoom:25%;" />





## 📊 Datasets Used

### Primary Dataset

- **Dataset**: VisDrone2019
- **Source**: http://aiskyeye.com/
- **Classes**: 10 object categories (person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others)
- **Images**: 8,629 (train), 548 (val), 1,610 (test)
- **Challenges**: High density, small objects, complex backgrounds

### Validation Dataset

- **Dataset**: DOTA (Dataset for Object Detection in Aerial Images)
- **Source**: https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip
- **Purpose**: Generalization ability validation
- **Classes**: 15 object categories
- **Characteristics**: Multi-scale, oriented objects

## ⚡ Performance Results

### Main Results on VisDrone2019

|         Model          | mAP50 | mAP50:95 | Params | GFLOPs |
| :--------------------: | :---: | :------: | :----: | :----: |
|     Faster  R-CNN      | 35.8  |   19.7   |   -    |   -    |
|      Sparse  DETR      | 42.5  |   27.3   |   -    | 121.0  |
| Vectorized  IOU-YOLOv5 | 44.6  |   26.6   |  19.3  |   -    |
|       UN-YOLOv5s       | 40.5  |   22.5   |   -    |  37.4  |
|      YOLOv7-tiny       | 35.0  |   18.5   |  6.04  |  13.3  |
|        YOLOv8s         | 40.4  |   24.0   |  11.1  |  28.7  |
|  Drone-YOLO  (large)   | 40.7  |    -     |  76.2  |   -    |
|        EBO-YOLO        | 41.1  |    -     |  8.0   |  20.4  |
|        EdgeYOLO        | 44.8  |    -     |  40.5  | 109.1  |
|       BDP-YOLOs        | 45.0  |   27.4   |  5.8   |  36.7  |
|       LRDS-YOLO        | 43.6  |   26.6   |  4.07  |  23.7  |
|        GF-YOLO         | 45.0  |   27.9   |  2.3   |  23.5  |



### Generalization Results on DOTA

|  Model   | *mAP*@0.5/% | *mAP*@0.5:0.95/% | Params/M |
| :------: | :---------: | :--------------: | :------: |
| YOLOv8n  |    40.9     |       24.6       |   3.0    |
| YOLOv11n |    39.7     |       24.6       |   2.6    |
| YOLOv8s  |    44.2     |       27.2       |   11.2   |
| YOLOv11s |    44.0     |       27.7       |   9.5    |
| GF-YOLO  |    46.6     |       28.6       |   2.3    |

## 🛠️ Implementation Details

### Training Configuration

```yaml
# Training settings for VisDrone2019
epochs=200
batch=16
imgsz=640

lr0=0.01,
lrf=0.01
momentum=0.937,
weight_decay=0.0005,
warmup_momentum=0.8,
warmup_bias_lr=0.1,
warmup_epochs=3.0,
save_period=50,
plots=True,
verbose=True,
cache=True,
scale=0.5,
fliplr=0.5,
mosaic=1.0,
mixup=0.0,
hsv_h=0.015,
hsv_s=0.7,
hsv_v=0.4,
```

## 🔬 Ablation Studies

### Component Analysis on VisDrone2019

|  Model  |  P2  | Delete P5 | SCE  | GHF  | DyHead | P    | R    | mAP50 | mAP50:95 | Params | GFLOPs |
| :-----: | :--: | :-------: | :--: | :--: | :----: | ---- | ---- | ----- | -------- | ------ | ------ |
| YOLOv8n |      |           |      |      |        | 44.5 | 32.8 | 33.0  | 19.3     | 3.0    | 8.9    |
|    A    |  √   |           |      |      |        | 48.7 | 35.6 | 37.1  | 22.3     | 2.9    | 12.4   |
|    B    |  √   |     √     |      |      |        | 47.2 | 35.7 | 36.6  | 21.9     | 1.0    | 10.6   |
|    C    |  √   |     √     |  √   |      |        | 50.3 | 37.1 | 38.7  | 23.4     | 1.0    | 12.8   |
|    D    |  √   |     √     |      |  √   |        | 48.8 | 36.2 | 37.9  | 22.8     | 1.1    | 12.5   |
|    E    |  √   |     √     |      |      |   √    | 52.4 | 39.9 | 41.7  | 25.3     | 2.0    | 17.4   |
|    F    |  √   |     √     |  √   |  √   |        | 51.6 | 40.4 | 42.0  | 25.4     | 1.2    | 16.2   |
|    G    |  √   |     √     |  √   |  √   |   √    | 54.5 | 42.0 | 45.0  | 27.9     | 2.3    | 23.5   |

## 🏃‍♂️ Quick Start




### Training

```bash
# Train GF-YOLO on VisDrone2019
python train.py --data configs/VisDrone.yaml --cfg ultralytics/cfg/models/v8.0/GF-YOLO.yaml --epochs 200

# Resume training
python train.py --data configs/VisDrone.yaml --cfg ultralytics/cfg/models/v8.0/GF-YOLO.yaml --resume 
```

## 📈 Visualization Results

### Detection Examples on VisDrone2019


<img src="Experimental Data\comparison.png" alt="comparison" style="zoom:50%;" />



<img src="Experimental Data\confusion_matrix_normalized.png" alt="confusion matrix" style="zoom:50%;" />







## 🔬 Technical Contributions

### Novel Architecture Design

1. **Scale-Aware Detection**: P2/P5 layer modification optimized for small objects
2. **Feature Enhancement**: SCE strategy improves shallow feature representation
3. **Adaptive Fusion**: GFF architecture with MFF and WFF modules
4. **Dynamic Attention**: Multi-level attention mechanism in detection head

### Computational Efficiency

- Maintains real-time performance suitable for edge devices
- Reduced parameter count through strategic layer removal
- Efficient feature fusion without significant computational overhead

## 📄 Data Availability Statement

**For Journal Submission**: The experimental results in this work are based on publicly available datasets:

- **VisDrone2019 Dataset**: Available at http://aiskyeye.com/ (primary evaluation dataset)
- **DOTA Dataset**: Available at https://captain-whu.github.io/DOTA/ (generalization validation)
- No new datasets were generated during this study
- All source code, model configurations, and trained weights are available in this repository: https://github.com/[username]/GF-YOLO
- Detailed experimental protocols and hyperparameters are provided for full reproducibility

## 📜 Citation

```bibtex
@article{gf_yolo_2025,
  title={Improved YOLOv8n Model for Efficient Aerial Object Detection},
  author={[Junkai Yi, Bobin Cui, Lingling Tan, Xuefeng Gao},
  journal={[Signal, Image and Video Processing]},
  year={2025},
  note={Under review}
}
```

## 🔗 Related Work

- [YOLOv8 Official Repository](https://github.com/ultralytics/ultralytics)
- [VisDrone Dataset](http://aiskyeye.com/)
- [DOTA Dataset](https://captain-whu.github.io/DOTA/)
- [Dynamic Head for Object Detection](https://arxiv.org/abs/2106.08322)
- [EfficientDet: Scalable and Efficient Object Detection ](https://arxiv.org/abs/1911.09070)

------

**Keywords**: UAV, Small targe, YOLO, Feature fusion
