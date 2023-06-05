# DeepStream-Yolo

NVIDIA DeepStream SDK 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 / 5.1  configuration for YOLO models

--------------------------------------------------------------------------------------------------
### Important: please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------

### Future updates

* DeepStream tutorials
* Updated INT8 calibration
* Support for segmentation models
* Support for classification models

### Improvements on this repository

* Support for INT8 calibration
* Support for non square models
* Models benchmarks
* Support for Darknet models (YOLOv4, etc) using cfg and weights conversion with GPU post-processing
* Support for YOLO-NAS, PPYOLOE+, PPYOLOE, DAMO-YOLO, YOLOX, YOLOR, YOLOv8, YOLOv7, YOLOv6 and YOLOv5 using ONNX conversion with GPU post-processing
* GPU bbox parser (it is slightly slower than CPU bbox parser on V100 GPU tests)
* **Support for DeepStream 5.1**
* **Custom ONNX model parser (`NvDsInferYoloCudaEngineGet`)**
* **Dynamic batch-size for Darknet and ONNX exported models**
* **INT8 calibration (PTQ) for Darknet and ONNX exported models**
* **New output structure (fix wrong output on DeepStream < 6.2) - it need to export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model**

##

### Getting started

* [Requirements](#requirements)
* [Suported models](#supported-models)
* [Benchmarks](docs/benchmarks.md)
* [dGPU installation](docs/dGPUInstalation.md)
* [Basic usage](#basic-usage)
* [Docker usage](#docker-usage)
* [NMS configuration](#nms-configuration)
* [INT8 calibration](docs/INT8Calibration.md)
* [YOLOv5 usage](docs/YOLOv5.md)
* [YOLOv6 usage](docs/YOLOv6.md)
* [YOLOv7 usage](docs/YOLOv7.md)
* [YOLOv8 usage](docs/YOLOv8.md)
* [YOLOR usage](docs/YOLOR.md)
* [YOLOX usage](docs/YOLOX.md)
* [DAMO-YOLO usage](docs/DAMOYOLO.md)
* [PP-YOLOE / PP-YOLOE+ usage](docs/PPYOLOE.md)
* [YOLO-NAS usage](docs/YOLONAS.md)
* [Using your custom model](docs/customModels.md)
* [Multiple YOLO GIEs](docs/multipleGIEs.md)

##

### Requirements

#### DeepStream 6.2 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.5 GA Update 1 (8.5.2.2)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 525.85.12 (Data center / Tesla series) / 525.105.17 (TITAN, GeForce RTX / GTX series and RTX / Quadro series)](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.2](https://developer.nvidia.com/deepstream-getting-started)
* [GStreamer 1.16.3](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.7 Update 1](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.4 GA (8.4.1.5)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 515.65.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1.1](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.6 Update 1](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.2 GA Update 4 (8.2.5.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 510.47.03](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.4 Update 1](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
* [TensorRT 8.0 GA (8.0.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 470.63.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.14.5](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 5.1 on x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
* [TensorRT 7.2.2](https://developer.nvidia.com/nvidia-tensorrt-7x-download)
* [NVIDIA Driver 460.32.03](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 5.1](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.14.5](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.2 on Jetson platform

* [JetPack 5.1.1 / 5.1](https://developer.nvidia.com/embedded/jetpack)
* [NVIDIA DeepStream SDK 6.2](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1.1 on Jetson platform

* [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack-sdk-502)
* [NVIDIA DeepStream SDK 6.1.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on Jetson platform

* [JetPack 5.0.1 DP](https://developer.nvidia.com/embedded/jetpack-sdk-501dp)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on Jetson platform

* [JetPack 4.6.2](https://developer.nvidia.com/embedded/jetpack-sdk-462)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 5.1 on Jetson platform

* [JetPack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive)
* [NVIDIA DeepStream SDK 5.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

##

### Suported models

* [Darknet](https://github.com/AlexeyAB/darknet)
* [MobileNet-YOLO](https://github.com/dog-qiuqiu/MobileNet-Yolo)
* [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv6](https://github.com/meituan/YOLOv6)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [YOLOR](https://github.com/WongKinYiu/yolor)
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)
* [PP-YOLOE / PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe)
* [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md)

##

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
```

#### 2. Download the `cfg` and `weights` files from [Darknet](https://github.com/AlexeyAB/darknet) repo to the DeepStream-Yolo folder

#### 3. Compile the lib

* DeepStream 6.2 on x86 platform

  ```
  CUDA_VER=11.8 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1.1 on x86 platform

  ```
  CUDA_VER=11.7 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 5.1 on x86 platform

  ```
  CUDA_VER=11.1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 / 5.1 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
  ```

#### 4. Edit the `config_infer_primary.txt` file according to your model (example for YOLOv4)

```
[property]
...
custom-network-config=yolov4.cfg
model-file=yolov4.weights
...
```

**NOTE**: By default, the dynamic batch-size is set. To use implicit batch-size, uncomment the line

```
...
force-implicit-batch-dim=1
...
```

#### 5. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: If you want to use YOLOv2 or YOLOv2-Tiny models, change the `deepstream_app_config.txt` file before run it

```
...
[primary-gie]
...
config-file=config_infer_primary_yoloV2.txt
...
```

##

### Docker usage

* x86 platform

  ```
  nvcr.io/nvidia/deepstream:6.2-devel
  nvcr.io/nvidia/deepstream:6.2-triton
  ```

* Jetson platform

  ```
  nvcr.io/nvidia/deepstream-l4t:6.2-samples
  nvcr.io/nvidia/deepstream-l4t:6.2-triton
  ```

  **NOTE**: To compile the `nvdsinfer_custom_impl_Yolo`, you need to install the g++ inside the container

  ```
  apt-get install build-essential
  ```

  **NOTE**: With DeepStream 6.2, the docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio track. Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features:
  
  ```
  /opt/nvidia/deepstream/deepstream/user_additional_install.sh
  ```

##

### NMS Configuration

To change the `nms-iou-threshold`, `pre-cluster-threshold` and `topk` values, modify the config_infer file

```
[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

**NOTE**: Make sure to set `cluster-mode=2` in the config_infer file.

##

### Extract metadata

You can get metadata from DeepStream using Python and C/C++. For C/C++, you can edit the `deepstream-app` or `deepstream-test` codes. For Python, your can install and edit [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

Basically, you need manipulate the `NvDsObjectMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsObjectMeta.html)) `and NvDsFrameMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsFrameMeta.html)) to get the label, position, etc. of bboxes.

##

My projects: https://www.youtube.com/MarcosLucianoTV
