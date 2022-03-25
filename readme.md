# DeepStream-Yolo

NVIDIA DeepStream SDK 6.0.1 configuration for YOLO models

### Future updates

* New documentation for multiple models
* DeepStream tutorials
* Native PP-YOLO support
* Dynamic batch-size

### Improvements on this repository

* Darknet CFG params parser (no need to edit nvdsparsebbox_Yolo.cpp or another file)
* Support for new_coords, beta_nms and scale_x_y params
* Support for new models
* Support for new layers
* Support for new activations
* Support for convolutional groups
* Support for INT8 calibration
* Support for non square models
* Support for reorg, implicit and channel layers (YOLOR)
* YOLOv5 6.0 / 6.1 native support
* YOLOR native support
* Models benchmarks (**outdated**)
* **GPU YOLO Decoder (moved from CPU to GPU to get better performance)** [#138](https://github.com/marcoslucianops/DeepStream-Yolo/issues/138)
* **Improved NMS** [#142](https://github.com/marcoslucianops/DeepStream-Yolo/issues/142)

##

### Getting started

* [Requirements](#requirements)
* [Tested models](#tested-models)
* [Benchmarks](#benchmarks)
* [dGPU installation](#dgpu-installation)
* [Basic usage](#basic-usage)
* [YOLOv5 usage](#yolov5-usage)
* [YOLOR usage](#yolor-usage)
* [INT8 calibration](#int8-calibration)
* [Using your custom model](docs/customModels.md)

##

### Requirements

#### x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.4](https://developer.nvidia.com/cuda-toolkit)
* [TensorRT 8.0 GA (8.0.1)](https://developer.nvidia.com/tensorrt)
* [cuDNN >= 8.2](https://developer.nvidia.com/cudnn)
* [NVIDIA Driver >= 470.63.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.0.1 (6.0)](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### Jetson platform

* [JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack)
* [NVIDIA DeepStream SDK 6.0.1 (6.0)](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

### For YOLOv5 and YOLOR

#### x86 platform

* [PyTorch >= 1.7.0](https://pytorch.org/get-started/locally/)

#### Jetson platform

* [PyTorch >= 1.7.0](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)

##

### Tested models

* [Darknet YOLO](https://github.com/AlexeyAB/darknet)
* [YOLOv5 6.0 / 6.1](https://github.com/ultralytics/yolov5)
* [YOLOR](https://github.com/WongKinYiu/yolor)
* [MobileNet-YOLO](https://github.com/dog-qiuqiu/MobileNet-Yolo)
* [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)

##

### Benchmarks

```
nms = 0.45 (changed to beta_nms when used in Darknet cfg file) / 0.6 (YOLOv5 and YOLOR models)
pre-cluster-threshold = 0.001 (mAP eval) / 0.25 (FPS measurement)
batch-size = 1
valid = val2017 (COCO) - 1000 random images for INT8 calibration
sample = 1920x1080 video
NOTE: Used maintain-aspect-ratio=1 in config_infer file for YOLOv4 (with letter_box=1), YOLOv5 and YOLOR models.
```

#### NVIDIA GTX 1050 4GB (Mobile)

##### YOLOR-CSP performance comparison

|                       | DeepStream | PyTorch |
|:---------------------:|:----------:|:-------:|
| FPS (without display) | 13.32      | 10.07   |
| FPS (with display)    | 12.63      | 9.41    |

##### YOLOv5n performance comparison

|                       | DeepStream | TensorRTx | Ultralytics |
|:---------------------:|:----------:|:---------:|:-----------:|
| FPS (without display) | 110.25     | 87.42     | 97.19       |
| FPS (with display)    | 105.62     | 73.07     | 50.37       |

<details><summary>More</summary>
<br>

| DeepStream         | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(without display) |
|:------------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:--------------------------:|
| YOLOR-P6           | FP32      | 1280       | 0.478        | 0.663   | 0.519    | 5.53                       |
| YOLOR-CSP-X*       | FP32      | 640        | 0.473        | 0.664   | 0.513    | 7.59                       |
| YOLOR-CSP-X        | FP32      | 640        | 0.470        | 0.661   | 0.507    | 7.52                       |
| YOLOR-CSP*         | FP32      | 640        | 0.459        | 0.652   | 0.496    | 13.28                      |
| YOLOR-CSP          | FP32      | 640        | 0.449        | 0.639   | 0.483    | 13.32                      |
| YOLOv5x6 6.0       | FP32      | 1280       | 0.504        | 0.681   | 0.547    | 2.22                       |
| YOLOv5l6 6.0       | FP32      | 1280       | 0.492        | 0.670   | 0.535    | 4.05                       |
| YOLOv5m6 6.0       | FP32      | 1280       | 0.463        | 0.642   | 0.504    | 7.54                       |
| YOLOv5s6 6.0       | FP32      | 1280       | 0.394        | 0.572   | 0.424    | 18.64                      |
| YOLOv5n6 6.0       | FP32      | 1280       | 0.294        | 0.452   | 0.314    | 26.94                      |
| YOLOv5x 6.0        | FP32      | 640        | 0.469        | 0.654   | 0.509    | 8.24                       |
| YOLOv5l 6.0        | FP32      | 640        | 0.450        | 0.634   | 0.487    | 14.96                      |
| YOLOv5m 6.0        | FP32      | 640        | 0.415        | 0.601   | 0.448    | 28.30                      |
| YOLOv5s 6.0        | FP32      | 640        | 0.334        | 0.516   | 0.355    | 63.55                      |
| YOLOv5n 6.0        | FP32      | 640        | 0.250        | 0.417   | 0.260    | 110.25                     |
| YOLOv4-P6          | FP32      | 1280       | 0.499        | 0.685   | 0.542    | 2.57                       |
| YOLOv4-P5          | FP32      | 896        | 0.472        | 0.659   | 0.513    | 5.48                       |
| YOLOv4-CSP-X-SWISH | FP32      | 640        | 0.473        | 0.664   | 0.513    | 7.51                       |
| YOLOv4-CSP-SWISH   | FP32      | 640        | 0.459        | 0.652   | 0.496    | 13.13                      |
| YOLOv4x-MISH       | FP32      | 640        | 0.459        | 0.650   | 0.495    | 7.53                       |
| YOLOv4-CSP         | FP32      | 640        | 0.440        | 0.632   | 0.474    | 13.19                      |
| YOLOv4             | FP32      | 608        | 0.498        | 0.740   | 0.549    | 12.18                      |
| YOLOv4-Tiny        | FP32      | 416        | 0.215        | 0.403   | 0.206    | 201.20                     |
| YOLOv3-SPP         | FP32      | 608        | 0.411        | 0.686   | 0.433    | 12.22                      |
| YOLOv3-Tiny-PRN    | FP32      | 416        | 0.167        | 0.382   | 0.125    | 277.14                     |
| YOLOv3             | FP32      | 608        | 0.377        | 0.672   | 0.385    | 12.51                      |
| YOLOv3-Tiny        | FP32      | 416        | 0.095        | 0.203   | 0.079    | 218.42                     |
| YOLOv2             | FP32      | 608        | 0.286        | 0.541   | 0.273    | 25.28                      |
| YOLOv2-Tiny        | FP32      | 416        | 0.102        | 0.258   | 0.061    | 231.36                     |

</details>

##

### dGPU installation

To install the DeepStream on dGPU (x86 platform), without docker, we need to do some steps to prepare the computer.

<details><summary>Open</summary>

#### 1. Disable Secure Boot in BIOS

<details><summary>If you are using a laptop with newer Intel/AMD processors, please update the kernel to newer version.</summary>

```
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-headers-5.11.0-051100_5.11.0-051100.202102142330_all.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-headers-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-image-unsigned-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-modules-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
sudo dpkg -i  *.deb
sudo reboot
```

</details>

#### 2. Install dependencies

```
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake
sudo apt-get install python3 python3-dev python3-pip
sudo apt install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4
sudo apt-get install linux-headers-$(uname -r)
```

**NOTE**: Install DKMS only if you are using the default Ubuntu kernel

```
sudo apt-get install dkms
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc.

```
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*tensorrt*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

#### 3. Disable Nouveau

```
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

* Add

```
blacklist nouveau
options nouveau modeset=0
```

* Run

```
sudo update-initramfs -u
```

#### 4. Reboot the computer

```
sudo reboot
```

#### 5. Download and install NVIDIA Driver without xconfig

* TITAN, GeForce RTX / GTX series and RTX / Quadro series

```
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/470.103.01/NVIDIA-Linux-x86_64-470.103.01.run
sudo sh NVIDIA-Linux-x86_64-470.103.01.run
```

* Data center / Tesla series

```
wget https://us.download.nvidia.com/tesla/470.103.01/NVIDIA-Linux-x86_64-470.103.01.run
sudo sh NVIDIA-Linux-x86_64-470.103.01.run
```

**NOTE**: Only if you are using default Ubuntu kernel, enable the DKMS during the installation.

#### 6. Download and install CUDA 11.4.3 without NVIDIA Driver

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda_11.4.3_470.82.01_linux.run
sudo sh cuda_11.4.3_470.82.01_linux.run
```

* Export environment variables

```
nano ~/.bashrc
```

* Add

```
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

* Run

```
source ~/.bashrc
sudo ldconfig
```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

#### 7. Download from [NVIDIA website](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the TensorRT 8.0 GA (8.0.1)

```
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda-repo.list
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-key add 7fa2af80.pub
sudo apt-get update
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626/7fa2af80.pub
sudo apt-get update
sudo apt-get install libnvinfer8=8.0.1-1+cuda11.3 libnvinfer-plugin8=8.0.1-1+cuda11.3 libnvparsers8=8.0.1-1+cuda11.3 libnvonnxparsers8=8.0.1-1+cuda11.3 libnvinfer-bin=8.0.1-1+cuda11.3 libnvinfer-dev=8.0.1-1+cuda11.3 libnvinfer-plugin-dev=8.0.1-1+cuda11.3 libnvparsers-dev=8.0.1-1+cuda11.3 libnvonnxparsers-dev=8.0.1-1+cuda11.3 libnvinfer-samples=8.0.1-1+cuda11.3 libnvinfer-doc=8.0.1-1+cuda11.3
```

#### 8. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-sdk) and install the DeepStream SDK 6.0.1 (6.0)

```
sudo apt-get install ./deepstream-6.0_6.0.1-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.4 /usr/local/cuda
```

#### 9. Reboot the computer

```
sudo reboot
```

</details>

##

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
```

#### 2. Download cfg and weights files from your model and move to DeepStream-Yolo folder

#### 3. Compile lib

* x86 platform

```
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform

```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

#### 4. Edit config_infer_primary.txt for your model (example for YOLOv4)

```
[property]
...
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
# YOLO cfg
custom-network-config=yolov4.cfg
# YOLO weights
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
# IOU threshold
nms-iou-threshold=0.6
# Socre threshold
pre-cluster-threshold=0.25
```

#### 5. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: If you want to use YOLOv2 or YOLOv2-Tiny models, change the deepstream_app_config.txt file before run it

```
...
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV2.txt
```

##

### YOLOv5 usage

#### 1. Copy gen_wts_yoloV5.py from DeepStream-Yolo/utils to [ultralytics/yolov5](https://github.com/ultralytics/yolov5) folder

#### 2. Open the ultralytics/yolov5 folder

#### 3. Download pt file from [ultralytics/yolov5](https://github.com/ultralytics/yolov5/releases/tag/v6.1) website (example for YOLOv5n)

```
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```

#### 4. Generate cfg and wts files (example for YOLOv5n)

```
python3 gen_wts_yoloV5.py -w yolov5n.pt
```

#### 5. Copy generated cfg and wts files to DeepStream-Yolo folder

#### 6. Open DeepStream-Yolo folder

#### 7. Compile lib

* x86 platform

```
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform

```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

#### 8. Edit config_infer_primary_yoloV5.txt for your model (example for YOLOv5n)

```
[property]
...
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
# CFG
custom-network-config=yolov5n.cfg
# WTS
model-file=yolov5n.wts
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
# IOU threshold
nms-iou-threshold=0.6
# Socre threshold
pre-cluster-threshold=0.25
```

#### 8. Change the deepstream_app_config.txt file

```
...
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yoloV5.txt
```

#### 9. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: For YOLOv5 P6 or custom models, check the gen_wts_yoloV5.py args and use them according to your model

* Input weights (.pt) file path **(required)**

```
-w or --weights
```

* Input cfg (.yaml) file path

```
-c or --yaml
```

* Model width **(default = 640 / 1280 [P6])**

```
-mw or --width
```

* Model height **(default = 640 / 1280 [P6])**

```
-mh or --height
```

* Model channels **(default = 3)**

```
-mc or --channels
```

* P6 model

```
--p6
```

##

### YOLOR usage

#### 1. Copy gen_wts_yolor.py from DeepStream-Yolo/utils to [yolor](https://github.com/WongKinYiu/yolor) folder

#### 2. Open the yolor folder

#### 3. Download pt file from [yolor](https://github.com/WongKinYiu/yolor) website

#### 4. Generate wts file (example for YOLOR-CSP)

```
python3 gen_wts_yolor.py -w yolor_csp.pt -c cfg/yolor_csp.cfg
```

#### 5. Copy cfg and generated wts files to DeepStream-Yolo folder

#### 6. Open DeepStream-Yolo folder

#### 7. Compile lib

* x86 platform

```
CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform

```
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

#### 8. Edit config_infer_primary_yolor.txt for your model (example for YOLOR-CSP)

```
[property]
...
# 0=RGB, 1=BGR, 2=GRAYSCALE
model-color-format=0
# CFG
custom-network-config=yolor_csp.cfg
# WTS
model-file=yolor_csp.wts
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
# IOU threshold
nms-iou-threshold=0.6
# Socre threshold
pre-cluster-threshold=0.25
```

#### 8. Change the deepstream_app_config.txt file

```
...
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
nvbuf-memory-type=0
config-file=config_infer_primary_yolor.txt
```

#### 9. Run

```
deepstream-app -c deepstream_app_config.txt
```

##

### INT8 calibration

#### 1. Install OpenCV

```
sudo apt-get install libopencv-dev
```

#### 2. Compile/recompile the nvdsinfer_custom_impl_Yolo lib with OpenCV support

* x86 platform

```
cd DeepStream-Yolo
CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform

```
cd DeepStream-Yolo
CUDA_VER=10.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
```

#### 3. For COCO dataset, download the [val2017](https://drive.google.com/file/d/1gbvfn7mcsGDRZ_luJwtITL-ru2kK99aK/view?usp=sharing), extract, and move to DeepStream-Yolo folder

##### Select 1000 random images from COCO dataset to run calibration

```
mkdir calibration
```

```
for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
    cp ${jpg} calibration/; \
done
```

##### Create the calibration.txt file with all selected images

```
realpath calibration/*jpg > calibration.txt
```

##### Set environment variables

```
export INT8_CALIB_IMG_PATH=calibration.txt
export INT8_CALIB_BATCH_SIZE=1
```

##### Change config_infer_primary.txt file

```
...
model-engine-file=model_b1_gpu0_fp32.engine
#int8-calib-file=calib.table
...
network-mode=0
...
```

* To

```
...
model-engine-file=model_b1_gpu0_int8.engine
int8-calib-file=calib.table
...
network-mode=1
...
```

##### Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: NVIDIA recommends at least 500 images to get a good accuracy. In this example I used 1000 images to get better accuracy (more images = more accuracy). Higher INT8_CALIB_BATCH_SIZE values will increase the accuracy and calibration speed. Set it according to you GPU memory. This process can take a long time.

##

### Extract metadata

You can get metadata from deepstream in Python and C++. For C++, you need edit deepstream-app or deepstream-test code. For Python your need install and edit [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

You need manipulate NvDsObjectMeta ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html)/[C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsObjectMeta.html)), NvDsFrameMeta ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html)/[C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsFrameMeta.html)) and NvOSD_RectParams ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvOSD/NvOSD_RectParams.html)/[C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvOSD__RectParams.html)) to get label, position, etc. of bboxes.

In C++ deepstream-app application, your code need be in analytics_done_buf_prob function.
In C++/Python deepstream-test application, your code need be in osd_sink_pad_buffer_probe/tiler_src_pad_buffer_probe function.

##

My projects: https://www.youtube.com/MarcosLucianoTV
