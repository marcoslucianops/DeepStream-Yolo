# Editing default model to your custom model
How to edit DeepStream files to your custom model

##

* [Requirements](#requirements)
* [Editing default model](#editing-default-model)
* [Compiling edited model](#compiling-edited-model)
* [Understanding and editing deepstream_app_config](#understanding-and-editing-deepstream_app_config)
* [Understanding and editing config_infer_primary](#understanding-and-editing-config_infer_primary)
* [Testing model](#testing-model)
* [Custom functions in your model](#custom-functions-in-your-model)

##

### Requirements
* [NVIDIA DeepStream SDK 5.0.1](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo Native](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native)
* [Pre-treined YOLO model](https://github.com/AlexeyAB/darknet)

##

### Editing default model
1. Run command
```
sudo chmod -R 777 /opt/nvidia/deepstream/deepstream-5.0/sources/
```

2. Download [my native folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/native), rename to yolo and move to your deepstream/sources folder.
3. Copy and remane your obj.names file to labels.txt to deepstream/sources/yolo directory
4. Copy your yolo.cfg and yolo.weights files to deepstream/sources/yolo directory.
5. Edit config_infer_primary.txt for your model
```
[property]
...
# CFG
custom-network-config=yolo.cfg
# Weights
model-file=yolo.weights
# Model labels file
labelfile-path=labels.txt
...
```

Note: if you want to use YOLOv2 or YOLOv2-Tiny models, change deepstream_app_config.txt
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

### Compiling edited model
1. Check your CUDA version (nvcc --version)
2. Go to deepstream/sources/yolo directory
3. Type command (example for CUDA 10.2 version):
```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Understanding and editing deepstream_app_config
To understand and edit deepstream_app_config.txt file, read the [DeepStream SDK Development Guide - Configuration Groups](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html#configuration-groups)

##

* Edit tiled-display

```
[tiled-display]
enable=1
# If you have 1 stream use 1/1 (rows/columns), if you have 4 streams use 2/2 or 4/1 or 1/4 (rows/columns)
rows=1
columns=1
# Resolution of tiled display
width=1280
height=720
gpu-id=0
nvbuf-memory-type=0
```

##

* Edit source

Example for 1 source:
```
[source0]
enable=1
# 1=Camera (V4L2), 2=URI, 3=MultiURI, 4=RTSP, 5=Camera (CSI; Jetson only)
type=3
# Stream URL
uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
# Number of sources copy (if > 1, you need edit rows/columns in tiled-display section and batch-size in streammux section and config_infer_primary.txt; need type=3 for more than 1 source)
num-sources=1
gpu-id=0
cudadec-memtype=0
```

Example for 1 duplcated source:
```
[source0]
enable=1
type=3
uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
num-sources=2
gpu-id=0
cudadec-memtype=0
```

Example for 2 sources:
```
[source0]
enable=1
type=3
uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
num-sources=1
gpu-id=0
cudadec-memtype=0

[source1]
enable=1
type=3
uri=rtsp://192.168.1.3/Streaming/Channels/101/httppreview
num-sources=1
gpu-id=0
cudadec-memtype=0
```

##

* Edit sink

Example for 1 source or 1 duplicated source:
```
[sink0]
enable=1
# 1=Fakesink, 2=EGL (nveglglessink), 3=Filesink, 4=RTSP, 5=Overlay (Jetson only)
type=2
# Indicates how fast the stream is to be rendered (0=As fast as possible, 1=Synchronously)
sync=0
# The ID of the source whose buffers this sink must use
source-id=0
gpu-id=0
nvbuf-memory-type=0
```

Example for 2 sources:
```
[sink0]
enable=1
type=2
sync=0
source-id=0
gpu-id=0
nvbuf-memory-type=0

[sink1]
enable=1
type=2
sync=0
source-id=1
gpu-id=0
nvbuf-memory-type=0
```

##

* Edit streammux

Example for 1 source:
```
[streammux]
gpu-id=0
# Boolean property to inform muxer that sources are live
live-source=1
# Number of sources
batch-size=1
# Time out in usec, to wait after the first buffer is available to push the batch even if the complete batch is not formed
batched-push-timeout=40000
# Resolution of streammux
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0
```

Example for 1 duplicated source or 2 sources:
```
[streammux]
gpu-id=0
live-source=0
batch-size=2
batched-push-timeout=40000
width=1920
height=1080
enable-padding=0
nvbuf-memory-type=0
```

##

* Edit primary-gie
```
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary.txt
```

* You can remove [tracker] section, if you don't use it.

##

### Understanding and editing config_infer_primary
To understand and edit config_infer_primary.txt file, read the [NVIDIA DeepStream Plugin Manual - Gst-nvinfer File Configuration Specifications](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications)

##

* Edit model-color-format accoding number of channels in yolo.cfg (1=GRAYSCALE, 3=RGB)

```
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
```

##

* Edit model-engine-file (example for batch-size=1 and network-mode=2)

```
model-engine-file=model_b1_gpu0_fp16.engine
```

##

* Edit batch-size

```
# Number of sources
batch-size=1
```

##

* Edit network-mode

```
# 0=FP32, 1=INT8, 2=FP16
network-mode=0
```

##

* Edit num-detected-classes according number of classes in yolo.cfg

```
num-detected-classes=80
```

##

* Edit network-type

```
# 0=Detector, 1=Classifier, 2=Segmentation
network-type=0
```

##

* Add/edit interval (FPS increase if > 0)

```
# Interval of detection
interval=0
```

##

* Change pre-cluster-threshold (optional)

```
[class-attrs-all]
# CONF_THRESH
pre-cluster-threshold=0.25
```

##

### Testing model

To run your custom YOLO model, use command
```
deepstream-app -c deepstream_app_config.txt
```
