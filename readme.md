# DeepStream-Yolo

NVIDIA DeepStream SDK 6.0 configuration for YOLO models

### Future updates (comming soon, stay tuned)

* New documentation for multiple models
* DeepStream tutorials
* Native PP-YOLO support
* Models benchmark
* GPU NMS
* Dynamic batch-size

### Improvements on this repository

* Darknet CFG params parser (it doesn't need to edit nvdsparsebbox_Yolo.cpp or another file for native models)
* Support for new_coords, beta_nms and scale_x_y params
* Support for new models
* Support for new layers
* Support for new activations
* Support for convolutional groups
* Support for INT8 calibration
* Support for non square models
* **Support for implicit and channel layers (YOLOR)**
* **YOLOv5 6.0 native support**
* **Initial YOLOR native support**

##

### Getting started

* [Requirements](#requirements)
* [Tested models](#tested-models)
* [dGPU installation](#dgpu-installation)
* [Basic usage](#basic-usage)
* [YOLOv5 usage](#yolov5-usage)
* [YOLOR usage](#yolor-usage)
* [INT8 calibration](#int8-calibration)
* [Using your custom model](docs/customModels.md)

##

### Requirements

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.4.3](https://developer.nvidia.com/cuda-toolkit)
* [TensorRT 8.0 GA (8.0.1)](https://developer.nvidia.com/tensorrt)
* [cuDNN >= 8.2](https://developer.nvidia.com/cudnn)
* [NVIDIA Driver >= 470.63.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.0](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

**For YOLOv5 and YOLOR**:

* [PyTorch >= 1.7.0](https://pytorch.org/get-started/locally/)

##

### Tested models
* [YOLOR-CSP](https://github.com/WongKinYiu/yolor) [[cfg]](https://raw.githubusercontent.com/WongKinYiu/yolor/main/cfg/yolor_csp.cfg) [[pt]](https://drive.google.com/file/d/1ZEqGy4kmZyD-Cj3tEFJcLSZenZBDGiyg/view?usp=sharing)
* [YOLOR-CSP*](https://github.com/WongKinYiu/yolor) [[cfg]](https://raw.githubusercontent.com/WongKinYiu/yolor/main/cfg/yolor_csp.cfg) [[pt]](https://drive.google.com/file/d/1OJKgIasELZYxkIjFoiqyn555bcmixUP2/view?usp=sharing)
* [YOLOR-CSP-X](https://github.com/WongKinYiu/yolor) [[cfg]](https://raw.githubusercontent.com/WongKinYiu/yolor/main/cfg/yolor_csp_x.cfg) [[pt]](https://drive.google.com/file/d/1L29rfIPNH1n910qQClGftknWpTBgAv6c/view?usp=sharing)
* [YOLOR-CSO-X*](https://github.com/WongKinYiu/yolor) [[cfg]](https://raw.githubusercontent.com/WongKinYiu/yolor/main/cfg/yolor_csp_x.cfg) [[pt]](https://drive.google.com/file/d/1NbMG3ivuBQ4S8kEhFJ0FIqOQXevGje_w/view?usp=sharing)
* [YOLOv5 6.0](https://github.com/ultralytics/yolov5) [[pt]](https://github.com/ultralytics/yolov5/releases/tag/v6.0)
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

**NOTE**: Install DKMS if you are using the default Ubuntu kernel

```
sudo apt-get install dkms
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc.

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

```
wget https://us.download.nvidia.com/tesla/470.82.01/NVIDIA-Linux-x86_64-470.82.01.run
sudo sh NVIDIA-Linux-x86_64-470.82.01.run
```

**NOTE**: If you are using default Ubuntu kernel, enable the DKMS during the installation. Else, you can skip this driver installation and install the NVIDIA driver from CUDA runfile (next step).

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

#### 8. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-sdk) and install the DeepStream SDK 6.0

```
sudo apt-get install ./deepstream-6.0_6.0.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
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

**NOTE**: The config_infer_primary.txt file uses cluster-mode=4 and NMS = 0.45 (via code) when beta_nms isn't available (when beta_nms is available, NMS = beta_nms), while the config_infer_primary_yoloV2.txt file uses cluster-mode=2 and nms-iou-threshold=0.45 to set NMS.

##

### YOLOv5 usage

#### 1. Copy gen_wts_yoloV5.py from DeepStream-Yolo/utils to [ultralytics/yolov5](https://github.com/ultralytics/yolov5) folder

#### 2. Open the ultralytics/yolov5 folder

#### 3. Download pt file from [ultralytics/yolov5](https://github.com/ultralytics/yolov5/releases/tag/v6.0) website (example for YOLOv5n)

```
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt
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
# CONF_THRESH
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

**NOTE**: For now, available only for YOLOR-CSP, YOLOR-CSP*, YOLOR-CSP-X and YOLOR-CSP-X*.

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
# CONF_THRESH
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

My projects: https://www.youtube.com/MarcosLucianoTV (new videos and tutorials comming soon)
