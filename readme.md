# DeepStream-Yolo
NVIDIA DeepStream SDK 5.0.1 configuration for YOLO models

##

### Improvements on this repository

* Darknet CFG params parser (not need to edit nvdsparsebbox_Yolo.cpp or another file for native models)
* Support to new_coords, beta_nms and scale_x_y params
* Support to new models not supported in official DeepStream SDK YOLO.
* Support to layers not supported in official DeepStream SDK YOLO.
* Support to activations not supported in official DeepStream SDK YOLO.
* Support to Convolutional groups

##

Tutorial
* [Configuring to your custom model](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/customModels.md)
* [Multiple YOLO inferences](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/multipleInferences.md)

Benchmark
* [mAP/FPS comparison between models](#mapfps-comparison-between-models)

TensorRT conversion
* [Native](#native-tensorrt-conversion) (tested models below)
    * YOLOv4x-Mish
    * YOLOv4-CSP
    * YOLOv4
    * YOLOv4-Tiny
    * YOLOv3-SSP
    * YOLOv3
    * YOLOv3-Tiny-PRN
    * YOLOv3-Tiny
    * YOLOv3-Lite
    * YOLOv3-Nano
    * YOLO-Fastest
    * YOLO-Fastest-XL
    * YOLOv2
    * YOLOv2-Tiny

* [External](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/YOLOv5.md)
    * YOLOv5

Request
* [Request native TensorRT conversion for your YOLO-based model](#request-native-tensorrt-conversion-for-your-yolo-based-model)

##

### Requirements
* [NVIDIA DeepStream SDK 5.0.1](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo Native](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native) (for Darknet YOLO based models)
* [DeepStream-Yolo External](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/external) (for PyTorch YOLOv5 based model)

##

### mAP/FPS comparison between models

DeepStream SDK YOLOv4: https://youtu.be/Qi_F_IYpuFQ

Darknet YOLOv4: https://youtu.be/AxJJ9fnJ7Xk

<details><summary>NVIDIA GTX 1050 (4GB Mobile)</summary>

```
CUDA 10.2
Driver 440.33
TensorRT 7.2.1
cuDNN 8.0.5
OpenCV 3.2.0 (libopencv-dev)
OpenCV Python 4.4.0 (opencv-python)
PyTorch 1.7.0
Torchvision 0.8.1
```

| TensorRT        | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with display) | FPS<br />(without display) |
|:---------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:-----------------------:|:--------------------------:|
| YOLOv5x         | FP32      | 608        | 0.406        | 0.562   | 0.441    | 7.91                    | 7.99                       |
| YOLOv5l         | FP32      | 608        | 0.385        | 0.540   | 0.419    | 12.82                   | 12.97                      |
| YOLOv5m         | FP32      | 608        | 0.354        | 0.507   | 0.388    | 25.09                   | 25.97                      |
| YOLOv5s         | FP32      | 608        | 0.281        | 0.430   | 0.307    | 52.02                   | 56.21                      |
| YOLOv4x-MISH    | FP32      | 640        | 0.454        | 0.644   | 0.491    | 7.45                    | 7.56                       |
| YOLOv4x-MISH    | FP32      | 608        | 0.450        | 0.644   | 0.482    | 7.93                    | 8.05                       |
| YOLOv4-CSP      | FP32      | 608        | 0.434        | 0.628   | 0.465    | 13.74                   | 14.11                      |
| YOLOv4-CSP      | FP32      | 512        | 0.427        | 0.618   | 0.459    | 21.69                   | 22.75                      |
| YOLOv4          | FP32      | 608        | 0.490        | 0.734   | 0.538    | 11.72                   | 12.09                      |
| YOLOv4          | FP32      | 512        | 0.484        | 0.725   | 0.533    | 19.00                   | 19.70                      |
| YOLOv4          | FP32      | 416        | 0.456        | 0.693   | 0.491    | 22.63                   | 23.81                      |
| YOLOv4          | FP32      | 320        | 0.400        | 0.623   | 0.424    | 32.46                   | 35.07                      |
| YOLOv3-SPP      | FP32      | 608        | 0.411        | 0.680   | 0.436    | 11.85                   | 12.12                      |
| YOLOv3          | FP32      | 608        | 0.374        | 0.654   | 0.387    | 12.00                   | 12.33                      |
| YOLOv3          | FP32      | 416        | 0.369        | 0.651   | 0.379    | 23.19                   | 24.55                      |
| YOLOv4-Tiny     | FP32      | 416        | 0.195        | 0.382   | 0.175    | 144.55                  | 176.31                     |
| YOLOv3-Tiny-PRN | FP32      | 416        | 0.168        | 0.369   | 0.130    | 181.71                  | 244.47                     |
| YOLOv3-Tiny     | FP32      | 416        | 0.165        | 0.357   | 0.128    | 154.19                  | 190.42                     |
| YOLOv3-Lite     | FP32      | 416        | 0.165        | 0.350   | 0.131    | 122.40                  | 146.19                     |
| YOLOv3-Lite     | FP32      | 320        | 0.155        | 0.324   | 0.128    | 163.76                  | 204.21                     |
| YOLOv3-Nano     | FP32      | 416        | 0.127        | 0.277   | 0.098    | 191.77                  | 264.59                     |
| YOLOv3-Nano     | FP32      | 320        | 0.122        | 0.258   | 0.099    | 207.04                  | 269.89                     |
| YOLO-Fastest    | FP32      | 416        | 0.092        | 0.213   | 0.062    | 174.26                  | 221.05                     |
| YOLO-Fastest    | FP32      | 320        | 0.090        | 0.201   | 0.068    | 199.48                  | 258.56                     |
| YOLO-FastestXL  | FP32      | 416        | 0.144        | 0.306   | 0.115    | 121.89                  | 145.13                     |
| YOLO-FastestXL  | FP32      | 320        | 0.136        | 0.279   | 0.117    | 162.65                  | 199.75                     |
| YOLOv2          | FP32      | 608        | 0.286        | 0.534   | 0.274    | 23.92                   | 25.47                      |
| YOLOv2-Tiny     | FP32      | 416        | 0.103        | 0.251   | 0.064    | 165.01                  | 203.02                     |

| Darknet         | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with display) | FPS<br />(without display) |
|:---------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:-----------------------:|:--------------------------:|
| YOLOv4x-MISH    | FP32      | 640        | 0.495        | 0.682   | 0.538    | 5.3                     | 5.5                        |
| YOLOv4x-MISH    | FP32      | 608        | 0.493        | 0.680   | 0.535    | 5.4                     | 5.6                        |
| YOLOv4-CSP      | FP32      | 608        | 0.473        | 0.661   | 0.515    | 9.2                     | 9.5                        |
| YOLOv4-CSP      | FP32      | 512        | 0.458        | 0.645   | 0.496    | 13.6                    | 14.0                       |
| YOLOv4          | FP32      | 608        | 0.513        | 0.748   | 0.574    | 7.3                     | 7.5                        |
| YOLOv4          | FP32      | 512        | 0.506        | 0.738   | 0.564    | 11.8                    | 12.3                       |
| YOLOv4          | FP32      | 416        | 0.479        | 0.709   | 0.527    | 15.4                    | 15.8                       |
| YOLOv4          | FP32      | 320        | 0.421        | 0.638   | 0.454    | 21.0                    | 21.7                       |
| YOLOv3-SPP      | FP32      | 608        | 0.432        | 0.701   | 0.465    | 6.9                     | 7.1                        |
| YOLOv3          | FP32      | 608        | 0.391        | 0.672   | 0.412    | 7.0                     | 7.3                        |
| YOLOv3          | FP32      | 416        | 0.384        | 0.668   | 0.402    | 16.3                    | 16.9                       |
| YOLOv4-Tiny     | FP32      | 416        | 0.203        | 0.388   | 0.189    | 68.0                    | 112.5                      |
| YOLOv3-Tiny-PRN | FP32      | 416        | 0.172        | 0.378   | 0.133    | 71.6                    | 143.9                      |
| YOLOv3-Tiny     | FP32      | 416        | 0.171        | 0.367   | 0.137    | 71.5                    | 117.9                      |
| YOLOv3-Lite     | FP32      | 416        | 0.169        | 0.349   | 0.144    | 53.8                    | 63.4                       |
| YOLOv3-Lite     | FP32      | 320        | 0.159        | 0.326   | 0.139    | 55.2                    | 97.5                       |
| YOLOv3-Nano     | FP32      | 416        | 0.129        | 0.275   | 0.102    | 58.0                    | 113.1                      |
| YOLOv3-Nano     | FP32      | 320        | 0.124        | 0.259   | 0.106    | 61.6                    | 156.8                      |
| YOLO-Fastest    | FP32      | 416        | 0.095        | 0.213   | 0.068    | 61.7                    | 104.1                      |
| YOLO-Fastest    | FP32      | 320        | 0.093        | 0.202   | 0.074    | 65.8                    | 143.3                      |
| YOLO-FastestXL  | FP32      | 416        | 0.148        | 0.308   | 0.125    | 62.0                    | 75.9                       |
| YOLO-FastestXL  | FP32      | 320        | 0.141        | 0.284   | 0.125    | 63.9                    | 112.3                      |
| YOLOv2          | FP32      | 608        | 0.297        | 0.548   | 0.291    | 12.1                    | 12.1                       |
| YOLOv2-Tiny     | FP32      | 416        | 0.105        | 0.255   | 0.068    | 34.5                    | 40.7                       |

| PyTorch | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with output) | FPS<br />(without output) |
|:-------:|:---------:|:----------:|:------------:|:-------:|:--------:|:----------------------:|:-------------------------:|
| YOLOv5x | FP32      | 608        | 0.487        | 0.676   | 0.527    | 8.25                   | 9.49                      |
| YOLOv5l | FP32      | 608        | 0.471        | 0.662   | 0.512    | 12.67                  | 15.77                     |
| YOLOv5m | FP32      | 608        | 0.439        | 0.631   | 0.474    | 18.13                  | 24.80                     |
| YOLOv5s | FP32      | 608        | 0.369        | 0.567   | 0.395    | 28.03                  | 49.52                     |

<br />

</details>

<details><summary>NVIDIA Jetson Nano (4GB)</summary>

```
JetPack 4.4.1
CUDA 10.2
TensorRT 7.1.3
cuDNN 8.0
OpenCV 4.1.1
```

| TensorRT        | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with display) | FPS<br />(without display) |
|:---------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:-----------------------:|:--------------------------:|
| YOLOv4          | FP32      | 416        | 0.462        | 0.694   | 0.503    | 2.97                   | 2.99                      |
| YOLOv4          | FP16      | 416        | 0.462        | 0.694   | 0.504    | 4.89                   | 4.96                      |
| YOLOv4          | FP32      | 320        | 0.407        | 0.625   | 0.434    |                    |                       |
| YOLOv4          | FP16      | 320        | 0.408        | 0.625   | 0.435    |                    |                       |
| YOLOv3          | FP32      | 416        | 0.370        | 0.664   | 0.379    |                    |                       |
| YOLOv3          | FP16      | 416        | 0.370        | 0.664   | 0.378    |                   |                       |
| YOLOv4-Tiny     | FP32      | 416        | 0.194        | 0.378   | 0.177    | 21.79                  | 23.23                      |
| YOLOv4-Tiny     | FP16      | 416        | 0.194        | 0.378   | 0.177    | 24.76                  | 26.18                      |
| YOLOv3-Tiny-PRN | FP32      | 416        | 0.163        | 0.375   | 0.120    | 23.79                  | 25.18                      |
| YOLOv3-Tiny-PRN | FP16      | 416        | 0.163        | 0.375   | 0.119    | 26.08                  | 27.96                      |
| YOLOv3-Tiny     | FP32      | 416        | 0.162        | 0.363   | 0.122    | 22.84                  | 24.28                      |
| YOLOv3-Tiny     | FP16      | 416        | 0.162        | 0.363   | 0.122    | 25.47                  | 27.18                      |

| Darknet         | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with display) | FPS<br />(without display) |
|:---------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:-----------------------:|:--------------------------:|
| YOLOv4          | FP32      | 416        |              |         |          |                         |                            |
| YOLOv4          | FP32      | 320        |              |         |          |                         |                            |
| YOLOv3          | FP32      | 416        |              |         |          |                         |                            |
| YOLOv4-Tiny     | FP32      | 416        |              |         |          |                         |                            |
| YOLOv3-Tiny-PRN | FP32      | 416        |              |         |          |                         |                            |
| YOLOv3-Tiny     | FP32      | 416        |              |         |          |                         |                            |
| YOLOv2          | FP32      | 608        |              |         |          |                         |                            |
| YOLOv2-Tiny     | FP32      | 416        |              |         |          |                         |                            |

| PyTorch | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with output) | FPS<br />(without output) |
|:-------:|:---------:|:----------:|:------------:|:-------:|:--------:|:----------------------:|:-------------------------:|
| YOLOv5s | FP32      | 416        |              |         |          |                        |                           |
| YOLOv5s | FP16      | 416        |              |         |          |                        |                           |

<br />

</details>

<br />

#### DeepStream settings

* General
```
width = 1920
height = 1080
maintain-aspect-ratio = 0
batch-size = 1
```

* Evaluate mAP
```
valid = val2017 (COCO)
nms-iou-threshold = 0.6
pre-cluster-threshold = 0.001 (CONF_THRESH)
```

* Evaluate FPS and Demo
```
nms-iou-threshold = 0.45 (NMS; changed to beta_nms when available)
pre-cluster-threshold = 0.25 (CONF_THRESH)
```

##

### Native TensorRT conversion
Download [my native folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native), rename to yolo and move to your deepstream/sources folder.

Download cfg and weights files from your model and move to deepstream/sources/yolo folder.

* [YOLOv4x-Mish](https://github.com/AlexeyAB/darknet) [[cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4x-mish.cfg)] [[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights)]
* [YOLOv4-CSP](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp) [[cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg)] [[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights)]
* [YOLOv4](https://github.com/AlexeyAB/darknet) [[cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg)] [[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)]
* [YOLOv4-Tiny](https://github.com/AlexeyAB/darknet) [[cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)] [[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)]
* [YOLOv3-SPP](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg)] [[weights](https://pjreddie.com/media/files/yolov3-spp.weights)]
* [YOLOv3](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)] [[weights](https://pjreddie.com/media/files/yolov3.weights)]
* [YOLOv3-Tiny-PRN](https://github.com/WongKinYiu/PartialResidualNetworks) [[cfg](https://raw.githubusercontent.com/WongKinYiu/PartialResidualNetworks/master/cfg/yolov3-tiny-prn.cfg)] [[weights](https://github.com/WongKinYiu/PartialResidualNetworks/raw/master/model/yolov3-tiny-prn.weights)]
* [YOLOv3-Tiny](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg)] [[weights](https://pjreddie.com/media/files/yolov3-tiny.weights)]
* [YOLOv3-Lite](https://github.com/dog-qiuqiu/MobileNet-Yolo) [[cfg](https://raw.githubusercontent.com/dog-qiuqiu/MobileNet-Yolo/master/MobileNetV2-YOLOv3-Lite/COCO/MobileNetV2-YOLOv3-Lite-coco.cfg)] [[weights](https://github.com/dog-qiuqiu/MobileNet-Yolo/raw/master/MobileNetV2-YOLOv3-Lite/COCO/MobileNetV2-YOLOv3-Lite-coco.weights)]
* [YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNet-Yolo) [[cfg](https://raw.githubusercontent.com/dog-qiuqiu/MobileNet-Yolo/master/MobileNetV2-YOLOv3-Nano/COCO/MobileNetV2-YOLOv3-Nano-coco.cfg)] [[weights](https://github.com/dog-qiuqiu/MobileNet-Yolo/raw/master/MobileNetV2-YOLOv3-Nano/COCO/MobileNetV2-YOLOv3-Nano-coco.weights)]
* [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) [[cfg](https://raw.githubusercontent.com/dog-qiuqiu/Yolo-Fastest/master/Yolo-Fastest/COCO/yolo-fastest.cfg)] [[weights](https://github.com/dog-qiuqiu/Yolo-Fastest/raw/master/Yolo-Fastest/COCO/yolo-fastest.weights)]
* [YOLO-Fastest-XL](https://github.com/dog-qiuqiu/Yolo-Fastest) [[cfg](https://raw.githubusercontent.com/dog-qiuqiu/Yolo-Fastest/master/Yolo-Fastest/COCO/yolo-fastest-xl.cfg)] [[weights](https://github.com/dog-qiuqiu/Yolo-Fastest/raw/master/Yolo-Fastest/COCO/yolo-fastest-xl.weights)]
* [YOLOv2](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg)] [[weights](https://pjreddie.com/media/files/yolov2.weights)]
* [YOLOv2-Tiny](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg)] [[weights](https://pjreddie.com/media/files/yolov2-tiny.weights)]


Compile
```
cd /opt/nvidia/deepstream/deepstream-5.0/sources/yolo
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

Edit config_infer_primary.txt for your model (example for YOLOv4)
```
[property]
...
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
# CFG
custom-network-config=yolov4.cfg
# Weights
model-file=yolov4.weights
# Generated TensorRT model (will be created if it doesn't exist)
model-engine-file=model_b1_gpu0_fp32.engine
# Model labels file
labelfile-path=labels.txt
# Batch size
batch-size=1
# 0=FP32, 1=INT8, 2=FP16 mode
network-mode=0
# Number of classes in label file
num-detected-classes=80
...
[class-attrs-all]
# CONF_THRESH
pre-cluster-threshold=0.25
```

Run
```
deepstream-app -c deepstream_app_config.txt
```

If you want to use YOLOv2 or YOLOv2-Tiny models, change, before run, deepstream_app_config.txt
```
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV2.txt
```

Note: config_infer_primary.txt uses cluster-mode=4 and NMS = 0.45 (via code) when beta_nms isn't available (when beta_nms is available, NMS = beta_nms), while config_infer_primary_yoloV2.txt uses cluster-mode=2 and nms-iou-threshold=0.45 to set NMS.

##

### Request native TensorRT conversion for your YOLO-based model
To request moded files for native TensorRT conversion to use in DeepStream SDK, send me the model cfg and weights files via Issues tab.

<br />

Note: If your model are listed in native tab, you can use [my native folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native) to run your model in DeepStream.

##

For commercial DeepStream SDK projects, contact me at email address available in GitHub.

My projects: https://www.youtube.com/MarcosLucianoTV