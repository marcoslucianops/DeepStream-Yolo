# DeepStream-Yolo

NVIDIA DeepStream SDK 6.1.1 / 6.1 / 6.0.1 / 6.0 configuration for YOLO models

### Future updates

* DeepStream tutorials
* YOLOX support
* YOLOv6 support
* Dynamic batch-size

### Improvements on this repository

* Darknet cfg params parser (no need to edit `nvdsparsebbox_Yolo.cpp` or other files)
* Support for `new_coords` and `scale_x_y` params
* Support for new models
* Support for new layers
* Support for new activations
* Support for convolutional groups
* Support for INT8 calibration
* Support for non square models
* New documentation for multiple models
* YOLOv5 support
* YOLOR support
* **GPU YOLO Decoder** [#138](https://github.com/marcoslucianops/DeepStream-Yolo/issues/138)
* **PP-YOLOE support**
* **YOLOv7 support**
* **Optimized NMS** [#142](https://github.com/marcoslucianops/DeepStream-Yolo/issues/142)
* **Models benchmarks**

##

### Getting started

* [Requirements](#requirements)
* [Suported models](#supported-models)
* [Benchmarks](#benchmarks)
* [dGPU installation](#dgpu-installation)
* [Basic usage](#basic-usage)
* [Docker usage](#docker-usage)
* [NMS configuration](#nms-configuration)
* [INT8 calibration](#int8-calibration)
* [YOLOv5 usage](docs/YOLOv5.md)
* [YOLOR usage](docs/YOLOR.md)
* [PP-YOLOE usage](docs/PPYOLOE.md)
* [YOLOv7 usage](docs/YOLOv7.md)
* [Using your custom model](docs/customModels.md)
* [Multiple YOLO GIEs](docs/multipleGIEs.md)

##

### Requirements

#### DeepStream 6.1.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.7 Update 1](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.4 GA (8.4.1.5)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 515.65.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1.1](https://developer.nvidia.com/deepstream-getting-started)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.6 Update 1](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.2 GA Update 4 (8.2.5.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 510.47.03](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/deepstream-getting-started)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.4 Update 1](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
* [TensorRT 8.0 GA (8.0.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver >= 470.63.01](https://www.nvidia.com.br/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.14.5](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1.1 on Jetson platform

* [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack)
* [NVIDIA DeepStream SDK 6.1.1](https://developer.nvidia.com/deepstream-sdk)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on Jetson platform

* [JetPack 5.0.1 DP](https://developer.nvidia.com/embedded/jetpack-sdk-501dp)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on Jetson platform

* [JetPack 4.6.2](https://developer.nvidia.com/embedded/jetpack-sdk-462)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

##

### Suported models

* [Darknet YOLO](https://github.com/AlexeyAB/darknet)
* [YOLOv5 >= 2.0](https://github.com/ultralytics/yolov5)
* [YOLOR](https://github.com/WongKinYiu/yolor)
* [PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [MobileNet-YOLO](https://github.com/dog-qiuqiu/MobileNet-Yolo)
* [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)

##

### Benchmarks

#### Config

```
board = NVIDIA Tesla V100 16GB (AWS: p3.2xlarge)
batch-size = 1
eval = val2017 (COCO)
sample = 1920x1080 video
```

**NOTE**: Used maintain-aspect-ratio=1 in config_infer file for Darknet (with letter_box=1) and PyTorch models.

#### NMS config

- Eval

```
nms-iou-threshold = 0.6 (Darknet) / 0.65 (PyTorch) / 0.7 (Paddle)
pre-cluster-threshold = 0.001
topk = 300
```

- Test

```
nms-iou-threshold = 0.45 / 0.7 (Paddle)
pre-cluster-threshold = 0.25
topk = 300
```

#### Results

**NOTE**: * = PyTorch

**NOTE**: ** = The YOLOv4 is trained with the trainvalno5k set, so the mAP is high on val2017 test

| DeepStream         | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(without display) |
|:------------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:--------------------------:|
| PP-YOLOE-x         | FP16      | 640        | 0.506        | 0.681   | 0.551    | 116.54                     |
| PP-YOLOE-l         | FP16      | 640        | 0.498        | 0.674   | 0.545    | 187.93                     |
| PP-YOLOE-m         | FP16      | 640        | 0.476        | 0.646   | 0.522    | 257.42                     |
| PP-YOLOE-s (400)   | FP16      | 640        | 0.422        | 0.589   | 0.463    | 465.23                     |
| YOLOv7-E6E         | FP16      | 1280       | 0.476        | 0.648   | 0.521    | 47.82                      |
| YOLOv7-D6          | FP16      | 1280       | 0.479        | 0.648   | 0.520    | 60.66                      |
| YOLOv7-E6          | FP16      | 1280       | 0.471        | 0.640   | 0.516    | 73.05                      |
| YOLOv7-W6          | FP16      | 1280       | 0.444        | 0.610   | 0.483    | 110.29                     |
| YOLOv7-X*          | FP16      | 640        | 0.496        | 0.679   | 0.536    | 162.31                     |
| YOLOv7*            | FP16      | 640        | 0.476        | 0.660   | 0.518    | 237.79                     |
| YOLOv7-Tiny Leaky* | FP16      | 640        | 0.345        | 0.516   | 0.372    | 611.36                     |
| YOLOv7-Tiny Leaky* | FP16      | 416        | 0.328        | 0.493   | 0.348    | 633.73                     |
| YOLOv5x6 6.1       | FP16      | 1280       | 0.508        | 0.683   | 0.554    | 54.88                      |
| YOLOv5l6 6.1       | FP16      | 1280       | 0.494        | 0.668   | 0.540    | 87.86                      |
| YOLOv5m6 6.1       | FP16      | 1280       | 0.469        | 0.644   | 0.514    | 142.68                     |
| YOLOv5s6 6.1       | FP16      | 1280       | 0.399        | 0.581   | 0.438    | 271.19                     |
| YOLOv5n6 6.1       | FP16      | 1280       | 0.317        | 0.487   | 0.344    | 392.20                     |
| YOLOv5x 6.1        | FP16      | 640        | 0.470        | 0.652   | 0.513    | 152.99                     |
| YOLOv5l 6.1        | FP16      | 640        | 0.454        | 0.636   | 0.496    | 247.60                     |
| YOLOv5m 6.1        | FP16      | 640        | 0.421        | 0.604   | 0.458    | 375.06                     |
| YOLOv5s 6.1        | FP16      | 640        | 0.344        | 0.528   | 0.371    | 602.44                     |
| YOLOv5n 6.1        | FP16      | 640        | 0.247        | 0.413   | 0.256    | 629.04                     |
| YOLOv4**           | FP16      | 608        | 0.497        | 0.739   | 0.549    | 206.23                     |
| YOLOv4-Tiny        | FP16      | 416        | 0.215        | 0.402   | 0.205    | 634.69                     |

##

### dGPU installation

To install the DeepStream on dGPU (x86 platform), without docker, we need to do some steps to prepare the computer.

<details><summary>DeepStream 6.1.1</summary>

#### 1. Disable Secure Boot in BIOS

#### 2. Install dependencies

```
sudo apt-get update
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev
sudo apt-get install linux-headers-$(uname -r)
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc (replace $CUDA_PATH to your CUDA path)

```
sudo nvidia-uninstall
sudo $CUDA_PATH/bin/cuda-uninstaller
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*tensorrt*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

#### 3. Install CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

#### 4. Download and install NVIDIA Driver

* TITAN, GeForce RTX / GTX series and RTX / Quadro series

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/515.65.01/NVIDIA-Linux-x86_64-515.65.01.run
  ```

* Data center / Tesla series

  ```
  wget https://us.download.nvidia.com/tesla/515.65.01/NVIDIA-Linux-x86_64-515.65.01.run
  ```

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

#### 5. Download and install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

#### 6. Download from [NVIDIA website](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the TensorRT

TensorRT 8.4 GA for Ubuntu 20.04 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6 and 11.7 DEB local repo Package

```
sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_amd64.deb 
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604/9a60d8bf.pub
sudo apt-get update
sudo apt-get install libnvinfer8=8.4.1-1+cuda11.6 libnvinfer-plugin8=8.4.1-1+cuda11.6 libnvparsers8=8.4.1-1+cuda11.6 libnvonnxparsers8=8.4.1-1+cuda11.6 libnvinfer-bin=8.4.1-1+cuda11.6 libnvinfer-dev=8.4.1-1+cuda11.6 libnvinfer-plugin-dev=8.4.1-1+cuda11.6 libnvparsers-dev=8.4.1-1+cuda11.6 libnvonnxparsers-dev=8.4.1-1+cuda11.6 libnvinfer-samples=8.4.1-1+cuda11.6 libcudnn8=8.4.1.50-1+cuda11.6 libcudnn8-dev=8.4.1.50-1+cuda11.6 python3-libnvinfer=8.4.1-1+cuda11.6 python3-libnvinfer-dev=8.4.1-1+cuda11.6
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* tensorrt
```

#### 7. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-getting-started) and install the DeepStream SDK

DeepStream 6.1.1 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-6.1_6.1.1-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.7 /usr/local/cuda
```

#### 8. Reboot the computer

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.1</summary>

#### 1. Disable Secure Boot in BIOS

#### 2. Install dependencies

```
sudo apt-get update
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev
sudo apt-get install linux-headers-$(uname -r)
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc (replace $CUDA_PATH to your CUDA path)

```
sudo nvidia-uninstall
sudo $CUDA_PATH/bin/cuda-uninstaller
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*tensorrt*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

#### 3. Install CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

#### 4. Download and install NVIDIA Driver

* TITAN, GeForce RTX / GTX series and RTX / Quadro series

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run
  ```

* Data center / Tesla series

  ```
  wget https://us.download.nvidia.com/tesla/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run
  ```

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

#### 5. Download and install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_510.47.03_linux.run
sudo sh cuda_11.6.1_510.47.03_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

#### 6. Download from [NVIDIA website](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the TensorRT

TensorRT 8.2 GA Update 4 for Ubuntu 20.04 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4 and 11.5 DEB local repo Package

```
sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.2.5.1-ga-20220505/82307095.pub
sudo apt-get update
sudo apt-get install libnvinfer8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 libnvparsers8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 libnvinfer-bin=8.2.5-1+cuda11.4 libnvinfer-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4 libnvparsers-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 libnvinfer-samples=8.2.5-1+cuda11.4 libnvinfer-doc=8.2.5-1+cuda11.4 libcudnn8-dev=8.4.0.27-1+cuda11.6 libcudnn8=8.4.0.27-1+cuda11.6
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* tensorrt
```

#### 7. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived) and install the DeepStream SDK

DeepStream 6.1 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-6.1_6.1.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.6 /usr/local/cuda
```

#### 8. Reboot the computer

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.0.1 / 6.0</summary>

#### 1. Disable Secure Boot in BIOS

<details><summary>If you are using a laptop with newer Intel/AMD processors and your Graphics in Settings->Details->About tab is llvmpipe, please update the kernel.</summary>

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
sudo apt-get update
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake
sudo apt-get install python3 python3-dev python3-pip
sudo apt-get install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4
sudo apt-get install linux-headers-$(uname -r)
```

**NOTE**: Install DKMS only if you are using the default Ubuntu kernel

```
sudo apt-get install dkms
```

**NOTE**: Purge all NVIDIA driver, CUDA, etc (replace $CUDA_PATH to your CUDA path)

```
sudo nvidia-uninstall
sudo $CUDA_PATH/bin/cuda-uninstaller
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*tensorrt*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

#### 3. Install CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

#### 4. Download and install NVIDIA Driver

* TITAN, GeForce RTX / GTX series and RTX / Quadro series

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
  ```

* Data center / Tesla series

  ```
  wget https://us.download.nvidia.com/tesla/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
  ```

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

#### 5. Download and install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
sudo sh cuda_11.4.1_470.57.02_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

#### 6. Download from [NVIDIA website](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and install the TensorRT

TensorRT 8.0.1 GA for Ubuntu 18.04 and CUDA 11.3 DEB local repo package

```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.3-trt8.0.1.6-ga-20210626/7fa2af80.pub
sudo apt-get update
sudo apt-get install libnvinfer8=8.0.1-1+cuda11.3 libnvinfer-plugin8=8.0.1-1+cuda11.3 libnvparsers8=8.0.1-1+cuda11.3 libnvonnxparsers8=8.0.1-1+cuda11.3 libnvinfer-bin=8.0.1-1+cuda11.3 libnvinfer-dev=8.0.1-1+cuda11.3 libnvinfer-plugin-dev=8.0.1-1+cuda11.3 libnvparsers-dev=8.0.1-1+cuda11.3 libnvonnxparsers-dev=8.0.1-1+cuda11.3 libnvinfer-samples=8.0.1-1+cuda11.3 libnvinfer-doc=8.0.1-1+cuda11.3 libcudnn8-dev=8.2.1.32-1+cuda11.3 libcudnn8=8.2.1.32-1+cuda11.3
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* tensorrt
```

#### 7. Download from [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived) and install the DeepStream SDK

* DeepStream 6.0.1 for Servers and Workstations (.deb)

  ```
  sudo apt-get install ./deepstream-6.0_6.0.1-1_amd64.deb
  ```

* DeepStream 6.0 for Servers and Workstations (.deb)

  ```
  sudo apt-get install ./deepstream-6.0_6.0.0-1_amd64.deb
  ```

* Run

  ```
  rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
  sudo ln -snf /usr/local/cuda-11.4 /usr/local/cuda
  ```

#### 8. Reboot the computer

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

#### 2. Download the `cfg` and `weights` files from [Darknet](https://github.com/AlexeyAB/darknet) repo to the DeepStream-Yolo folder

#### 3. Compile the lib

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

* DeepStream 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

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

#### 5. Run

```
deepstream-app -c deepstream_app_config.txt
```

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
  nvcr.io/nvidia/deepstream:6.1.1-devel
  nvcr.io/nvidia/deepstream:6.1.1-triton
  ```

* Jetson platform

  ```
  nvcr.io/nvidia/deepstream-l4t:6.1.1-samples
  nvcr.io/nvidia/deepstream-l4t:6.1.1-triton
  ```

  **NOTE**: To compile the `nvdsinfer_custom_impl_Yolo`, you need to install the g++ inside the container

  ```
  apt-get install build-essential
  ```

  **NOTE**: With DeepStream 6.1.1, the docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio track. Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features:
  
  ```
  /opt/nvidia/deepstream/deepstream/user_additional_install.sh
  ```

  **NOTE**: With DeepStream 6.1, the container image missed to include certain header files that will be available on host machine with Compute libraries installed from Jetpack. To mount the headers, use:

  ```
  -v /usr/include/aarch64-linux-gnu/NvInfer.h:/usr/include/aarch64-linux-gnu/NvInfer.h -v /usr/include/aarch64-linux-gnu/NvInferLegacyDims.h:/usr/include/aarch64-linux-gnu/NvInferLegacyDims.h -v /usr/include/aarch64-linux-gnu/NvInferRuntimeCommon.h:/usr/include/aarch64-linux-gnu/NvInferRuntimeCommon.h -v /usr/include/aarch64-linux-gnu/NvInferVersion.h:/usr/include/aarch64-linux-gnu/NvInferVersion.h -v /usr/include/aarch64-linux-gnu/NvInferRuntime.h:/usr/include/aarch64-linux-gnu/NvInferRuntime.h -v /usr/include/aarch64-linux-gnu/NvInferImpl.h:/usr/include/aarch64-linux-gnu/NvInferImpl.h -v /usr/include/aarch64-linux-gnu/NvCaffeParser.h:/usr/include/aarch64-linux-gnu/NvCaffeParser.h -v /usr/include/aarch64-linux-gnu/NvUffParser.h:/usr/include/aarch64-linux-gnu/NvUffParser.h -v /usr/include/aarch64-linux-gnu/NvInferPlugin.h:/usr/include/aarch64-linux-gnu/NvInferPlugin.h -v /usr/include/aarch64-linux-gnu/NvInferPluginUtils.h:/usr/include/aarch64-linux-gnu/NvInferPluginUtils.h -v /usr/local/cuda/:/usr/local/cuda/
  ```

  <details>
  <summary>Example</summary>

  ```
  sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-6.1 -v /tmp/.X11-unix/:/tmp/.X11-unix -v /usr/include/aarch64-linux-gnu/NvInfer.h:/usr/include/aarch64-linux-gnu/NvInfer.h -v /usr/include/aarch64-linux-gnu/NvInferLegacyDims.h:/usr/include/aarch64-linux-gnu/NvInferLegacyDims.h -v /usr/include/aarch64-linux-gnu/NvInferRuntimeCommon.h:/usr/include/aarch64-linux-gnu/NvInferRuntimeCommon.h -v /usr/include/aarch64-linux-gnu/NvInferVersion.h:/usr/include/aarch64-linux-gnu/NvInferVersion.h -v /usr/include/aarch64-linux-gnu/NvInferRuntime.h:/usr/include/aarch64-linux-gnu/NvInferRuntime.h -v /usr/include/aarch64-linux-gnu/NvInferImpl.h:/usr/include/aarch64-linux-gnu/NvInferImpl.h -v /usr/include/aarch64-linux-gnu/NvCaffeParser.h:/usr/include/aarch64-linux-gnu/NvCaffeParser.h -v /usr/include/aarch64-linux-gnu/NvUffParser.h:/usr/include/aarch64-linux-gnu/NvUffParser.h -v /usr/include/aarch64-linux-gnu/NvInferPlugin.h:/usr/include/aarch64-linux-gnu/NvInferPlugin.h -v /usr/include/aarch64-linux-gnu/NvInferPluginUtils.h:/usr/include/aarch64-linux-gnu/NvInferPluginUtils.h -v /usr/local/cuda/:/usr/local/cuda/ nvcr.io/nvidia/deepstream-l4t:6.1-samples
  ```
  </details>

##

### NMS Configuration

To change the `nms-iou-threshold`, `pre-cluster-threshold` and `topk` values, modify the config_infer file and regenerate the model engine file

```
[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

**NOTE**: It is important to regenerate the engine to get the max detection speed based on `pre-cluster-threshold` you set.

**NOTE**: Lower `topk` values will result in more performance.

**NOTE**: Make sure to set `cluster-mode=2` in the config_infer file.

##

### INT8 calibration

#### 1. Install OpenCV

```
sudo apt-get install libopencv-dev
```

#### 2. Compile/recompile the `nvdsinfer_custom_impl_Yolo` lib with OpenCV support

* DeepStream 6.1.1 on x86 platform

  ```
  CUDA_VER=11.7 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 OPENCV=1 make -C nvdsinfer_custom_impl_Yolo
  ```

#### 3. For COCO dataset, download the [val2017](https://drive.google.com/file/d/1gbvfn7mcsGDRZ_luJwtITL-ru2kK99aK/view?usp=sharing), extract, and move to DeepStream-Yolo folder

* Select 1000 random images from COCO dataset to run calibration

  ```
  mkdir calibration
  ```

  ```
  for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do \
      cp ${jpg} calibration/; \
  done
  ```

* Create the `calibration.txt` file with all selected images

  ```
  realpath calibration/*jpg > calibration.txt
  ```

* Set environment variables

  ```
  export INT8_CALIB_IMG_PATH=calibration.txt
  export INT8_CALIB_BATCH_SIZE=1
  ```

* Edit the `config_infer` file

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

* Run

  ```
  deepstream-app -c deepstream_app_config.txt
  ```

**NOTE**: NVIDIA recommends at least 500 images to get a good accuracy. On this example, I used 1000 images to get better accuracy (more images = more accuracy). Higher `INT8_CALIB_BATCH_SIZE` values will result in more accuracy and faster calibration speed. Set it according to you GPU memory. This process can take a long time.

##

### Extract metadata

You can get metadata from DeepStream using Python and C/C++. For C/C++, you can edit the `deepstream-app` or `deepstream-test` codes. For Python, your can install and edit [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

Basically, you need manipulate the `NvDsObjectMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsObjectMeta.html)) `and NvDsFrameMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsFrameMeta.html)) to get the label, position, etc. of bboxes.

##

My projects: https://www.youtube.com/MarcosLucianoTV
