# DeepStream-Yolo
NVIDIA DeepStream SDK 5.1 configuration for YOLO models

##

### Improvements on this repository

* Darknet CFG params parser (not need to edit nvdsparsebbox_Yolo.cpp or another file for native models)
* Support for new_coords, beta_nms and scale_x_y params
* Support for new models not supported in official DeepStream SDK YOLO.
* Support for layers not supported in official DeepStream SDK YOLO.
* Support for activations not supported in official DeepStream SDK YOLO.
* Support for Convolutional groups
* **Support for INT8 calibration** (not available for YOLOv5 models)
* **Support for non square models**

##

Tutorial
* [Basic usage](#basic-usage)
* [INT8 calibration](#int8-calibration)
* [Configuring to your custom model](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/customModels.md)
* [Multiple YOLO inferences](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/multipleInferences.md)

TensorRT conversion
* Native (tested models below)
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
    * [YOLO-Fastest 1.1](https://github.com/dog-qiuqiu/Yolo-Fastest) [[cfg](https://raw.githubusercontent.com/dog-qiuqiu/Yolo-Fastest/master/ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1-xl.cfg)] [[weights](https://github.com/dog-qiuqiu/Yolo-Fastest/raw/master/ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1-xl.weights)]
    * [YOLO-Fastest-XL 1.1](https://github.com/dog-qiuqiu/Yolo-Fastest) [[cfg](https://raw.githubusercontent.com/dog-qiuqiu/Yolo-Fastest/master/ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1.cfg)] [[weights](https://github.com/dog-qiuqiu/Yolo-Fastest/raw/master/ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1.weights)]
    * [YOLOv2](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg)] [[weights](https://pjreddie.com/media/files/yolov2.weights)]
    * [YOLOv2-Tiny](https://github.com/pjreddie/darknet) [[cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg)] [[weights](https://pjreddie.com/media/files/yolov2-tiny.weights)]

* External
    * [YOLOv5 5.0](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/YOLOv5-5.0.md)
    * [YOLOv5 4.0](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/YOLOv5-4.0.md)
    * [YOLOv5 3.X (3.0/3.1)](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/YOLOv5-3.X.md)

Benchmark
* [mAP/FPS comparison between models](#mapfps-comparison-between-models)

##

### Requirements
* [NVIDIA DeepStream SDK 5.1](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo Native](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native) (for Darknet YOLO based models)
* [DeepStream-Yolo External](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/external) (for PyTorch YOLOv5 based model)

##

### Basic usage

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo/native
```

Download cfg and weights files from your model and move to DeepStream-Yolo/native folder

Compile

* x86 platform
```
CUDA_VER=11.1 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform
```
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

### INT8 calibration

Install OpenCV
```
sudo apt-get install libopencv-dev
```

Compile/recompile the nvdsinfer_custom_impl_Yolo lib with OpenCV support

* x86 platform
```
cd DeepStream-Yolo/native
CUDA_VER=11.1 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform
```
cd DeepStream-Yolo/native
CUDA_VER=10.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
```

For COCO dataset, download the [val2017](https://drive.google.com/file/d/1gbvfn7mcsGDRZ_luJwtITL-ru2kK99aK/view?usp=sharing), extract, and move to DeepStream-Yolo/native folder

Select 1000 random images from COCO dataset to run calibration
```
mkdir calibration
for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
    cp val2017/${jpg} calibration/; \
done
```

Create the calibration.txt file with all selected images
```
realpath calibration/*jpg > calibration.txt
```

Set environment variables
```
export INT8_CALIB_IMG_PATH=calibration.txt
export INT8_CALIB_BATCH_SIZE=1
```

Change config_infer_primary.txt file
```
...
model-engine-file=model_b1_gpu0_fp32.engine
#int8-calib-file=calib.table
...
network-mode=0
...
```
To
```
...
model-engine-file=model_b1_gpu0_int8.engine
int8-calib-file=calib.table
...
network-mode=1
...
```

Run
```
deepstream-app -c deepstream_app_config.txt
```

Note: NVIDIA recommends at least 500 images to get a good accuracy. In this example I used 1000 images to get better accuracy (more images = more accuracy). Higher INT8_CALIB_BATCH_SIZE values will increase the accuracy and calibration speed. Set it according to you GPU memory. This process can take a long time. The calibration isn't available for YOLOv5 models.

##

### mAP/FPS comparison between models

<details><summary>Open</summary>

```
valid = val2017 (COCO)
NMS = 0.45 (changed to beta_nms when used in Darknet cfg file) / 0.6 (YOLOv5 models)
pre-cluster-threshold = 0.001 (mAP eval) / 0.25 (FPS measurement)
batch-size = 1
FPS measurement display width = 1920
FPS measurement display height = 1080
NOTE: Used NVIDIA GTX 1050 (4GB Mobile) for evaluate. Used maintain-aspect-ratio=1 in config_infer file for YOLOv4 (with letter_box=1) and YOLOv5 models. For INT8 calibration, was used 1000 random images from val2017 (COCO) and INT8_CALIB_BATCH_SIZE=1.
```

| TensorRT        | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(with display) | FPS<br />(without display) |
|:---------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:-----------------------:|:--------------------------:|
| YOLOv5x 5.0     | FP32      | 640        | 0.        | 0.   | 0.    | .                    | .                       |
| YOLOv5l 5.0     | FP32      | 640        | 0.        | 0.   | 0.    | .                   | .                      |
| YOLOv5m 5.0     | FP32      | 640        | 0.        | 0.   | 0.    | .                   | .                      |
| YOLOv5s 5.0     | FP32      | 640        | 0.        | 0.   | 0.    | .                   | .                      |
| YOLOv5s 5.0     | FP32      | 416        | 0.        | 0.   | 0.    | .                   | .                      |
| YOLOv4x-MISH    | FP32      | 640        | 0.461        | 0.649   | 0.499    | .                    | .                       |
| YOLOv4x-MISH    | **INT8**  | 640        | 0.443        | 0.629   | 0.479    | .                    | .                       |
| YOLOv4x-MISH    | FP32      | 608        | 0.461        | 0.650   | 0.496    | .                    | .                       |
| YOLOv4-CSP      | FP32      | 640        | 0.443        | 0.632   | 0.477    | .                   | .                      |
| YOLOv4-CSP      | FP32      | 608        | 0.443        | 0.632   | 0.477    | .                   | .                      |
| YOLOv4-CSP      | FP32      | 512        | 0.437        | 0.625   | 0.471    | .                   | .                      |
| YOLOv4-CSP      | **INT8**  | 512        | 0.414        | 0.601   | 0.447    | .                    | .                       |
| YOLOv4          | FP32      | 640        | 0.492        | 0.729   | 0.547    | .                   | .                      |
| YOLOv4          | FP32      | 608        | 0.499        | 0.739   | 0.551    | .                   | .                      |
| YOLOv4          | **INT8**  | 608        | 0.483        | 0.728   | 0.534    | .                    | .                       |
| YOLOv4          | FP32      | 512        | 0.492        | 0.730   | 0.542    | .                   | .                      |
| YOLOv4          | FP32      | 416        | 0.468        | 0.702   | 0.507    | .                   | .                      |
| YOLOv3-SPP      | FP32      | 608        | 0.412        | 0.687   | 0.434    | .                   | .                      |
| YOLOv3          | FP32      | 608        | 0.378        | 0.674   | 0.389    | .                   | .                      |
| YOLOv3          | **INT8**  | 608        | 0.381        | 0.677   | 0.388    | .                    | .                       |
| YOLOv3          | FP32      | 416        | 0.373        | 0.669   | 0.379    | .                   | .                      |
| YOLOv2          | FP32      | 608        | 0.211        | 0.365   | 0.220    | .                   | .                      |
| YOLOv2          | FP32      | 416        | 0.207        | 0.362   | 0.211    | .                   | .                      |
| YOLOv4-Tiny     | FP32      | 416        | 0.216        | 0.403   | 0.207    | .                  | .                     |
| YOLOv4-Tiny     | **INT8**  | 416        | 0.203        | 0.385   | 0.192    | .                  | .                     |
| YOLOv3-Tiny-PRN | FP32      | 416        | 0.168        | 0.381   | 0.126    | .                  | .                     |
| YOLOv3-Tiny-PRN | **INT8**  | 416        | 0.155        | 0.358   | 0.113    | .                  | .                     |
| YOLOv3-Tiny     | FP32      | 416        | 0.096        | 0.203   | 0.080    | .                  | .                     |
| YOLOv2-Tiny     | FP32      | 416        | 0.084        | 0.194   | 0.062    | .                  | .                     |
| YOLOv3-Lite     | FP32      | 416        | 0.169        | 0.356   | 0.137    | .                  | .                     |
| YOLOv3-Lite     | FP32      | 320        | 0.158        | 0.328   | 0.132    | .                  | .                     |
| YOLOv3-Nano     | FP32      | 416        | 0.128        | 0.278   | 0.099    | .                  | .                     |
| YOLOv3-Nano     | FP32      | 320        | 0.122        | 0.260   | 0.099    | .                  | .                     |
| YOLO-Fastest-XL | FP32      | 416        | 0.160        | 0.342   | 0.130    | .                  | .                     |
| YOLO-Fastest-XL | FP32      | 320        | 0.158        | 0.329   | 0.135    | .                  | .                     |
| YOLO-Fastest    | FP32      | 416        | 0.101        | 0.230   | 0.072    | .                  | .                     |
| YOLO-Fastest    | FP32      | 320        | 0.102        | 0.232   | 0.073    | .                  | .                     |

</details>

##

### Extract metadata

You can get metadata from deepstream in Python and C++. For C++, you need edit deepstream-app or deepstream-test code. For Python your need install and edit [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

You need manipulate NvDsObjectMeta ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html)/[C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/Meta/_NvDsObjectMeta.html)), NvDsFrameMeta ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html)/[C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/Meta/_NvDsFrameMeta.html)) and NvOSD_RectParams ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsOSD/NvOSD_RectParams.html)/[C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/OSD/Data_Structures/_NvOSD_FrameRectParams.html)) to get label, position, etc. of bboxs.

In C++ deepstream-app application, your code need be in analytics_done_buf_prob function.
In C++/Python deepstream-test application, your code need be in osd_sink_pad_buffer_probe/tiler_src_pad_buffer_probe function.

Python is slightly slower than C (about 5-10%).

##

This code is open-source. You can use as you want. :)

If you want me to create commercial DeepStream SDK projects for you, contact me at email address available in GitHub.

My projects: https://www.youtube.com/MarcosLucianoTV
