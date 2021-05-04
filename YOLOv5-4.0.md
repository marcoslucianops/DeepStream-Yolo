# YOLOv5
NVIDIA DeepStream SDK 5.1 configuration for YOLOv5 4.0 models

Thanks [wang-xinyu](https://github.com/wang-xinyu/tensorrtx) and [Ultralytics](https://github.com/ultralytics/yolov5)

##

* [Requirements](#requirements)
* [Convert PyTorch model to wts file](#convert-pytorch-model-to-wts-file)
* [Convert wts file to TensorRT model](#convert-wts-file-to-tensorrt-model)
* [Compile nvdsinfer_custom_impl_Yolo](#compile-nvdsinfer_custom_impl_yolo)
* [Testing model](#testing-model)

##

### Requirements
* [TensorRTX](https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/install.md)

* [Ultralytics](https://github.com/ultralytics/yolov5/blob/v4.0/requirements.txt)

* Matplotlib (for Jetson plataform)
```
sudo apt-get install python3-matplotlib
```

* PyTorch (for Jetson plataform)
```
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl
```

* TorchVision (for Jetson platform)
```
git clone -b v0.7.0 https://github.com/pytorch/vision torchvision
sudo apt-get install libjpeg-dev zlib1g-dev python3-pip
cd torchvision
export BUILD_VERSION=0.7.0
sudo python3 setup.py install
```

##

### Convert PyTorch model to wts file
1. Download repositories
```
git clone -b yolov5-v4.0 https://github.com/wang-xinyu/tensorrtx.git
git clone -b v4.0 https://github.com/ultralytics/yolov5.git
```

2. Download latest YoloV5 (YOLOv5s, YOLOv5m, YOLOv5l or YOLOv5x) weights to yolov5 folder (example for YOLOv5s)
```
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt -P yolov5/weights
```

3. Copy gen_wts.py file (from tensorrtx/yolov5 folder) to yolov5 (ultralytics) folder
```
cp tensorrtx/yolov5/gen_wts.py yolov5/gen_wts.py
```

4. Generate wts file
```
cd yolov5
python3 gen_wts.py
```

yolov5s.wts file will be generated in yolov5 folder

##

### Convert wts file to TensorRT model
1. Build tensorrtx/yolov5
```
cd tensorrtx/yolov5
mkdir build
cd build
cmake ..
make
```

2. Move generated yolov5s.wts file to tensorrtx/yolov5 folder (example for YOLOv5s)
```
cp yolov5/yolov5s.wts tensorrtx/yolov5/build/yolov5s.wts
```

3. Convert to TensorRT model (yolov5s.engine file will be generated in tensorrtx/yolov5/build folder)
```
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
```

4. Create a custom yolo folder and copy generated file (example for YOLOv5s)
```
mkdir /opt/nvidia/deepstream/deepstream-5.1/sources/yolo
cp yolov5s.engine /opt/nvidia/deepstream/deepstream-5.1/sources/yolo/yolov5s.engine
```

<br />

Note: by default, yolov5 script generate model with batch size = 1 and FP16 mode.
```
#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
```
Edit yolov5.cpp file before compile if you want to change this parameters.

##

### Compile nvdsinfer_custom_impl_Yolo
1. Run command
```
sudo chmod -R 777 /opt/nvidia/deepstream/deepstream-5.1/sources/
```

2. Donwload [my external/yolov5-4.0 folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/external/yolov5-4.0) and move files to created yolo folder

3. Compile lib

* x86 platform
```
cd /opt/nvidia/deepstream/deepstream-5.1/sources/yolo
CUDA_VER=11.1 make -C nvdsinfer_custom_impl_Yolo
```

* Jetson platform
```
cd /opt/nvidia/deepstream/deepstream-5.1/sources/yolo
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Testing model
Use my edited [deepstream_app_config.txt](https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/master/external/yolov5-4.0/deepstream_app_config.txt) and [config_infer_primary.txt](https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/master/external/yolov5-4.0/config_infer_primary.txt) files available in [my external/yolov5-4.0 folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/external/yolov5-4.0)

Run command
```
deepstream-app -c deepstream_app_config.txt
```

<br />

Note: based on selected model, edit config_infer_primary.txt file

For example, if you using YOLOv5x

```
model-engine-file=yolov5s.engine
```

to

```
model-engine-file=yolov5x.engine
```

##

To change NMS_THRESH, edit nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp file and recompile

```
#define kNMS_THRESH 0.45
```

To change CONF_THRESH, edit config_infer_primary.txt file

```
[class-attrs-all]
pre-cluster-threshold=0.25
```
