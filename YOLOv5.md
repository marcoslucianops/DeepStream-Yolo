# YOLOv5
NVIDIA DeepStream SDK 5.0.1 configuration for YOLOv5 models

Thanks [DanaHan](https://github.com/DanaHan/Yolov5-in-Deepstream-5.0), [wang-xinyu](https://github.com/wang-xinyu/tensorrtx) and [Ultralytics](https://github.com/ultralytics/yolov5)

Supported version: YOLOv5 3.0/3.1

##

* [Requirements](#requirements)
* [Convert PyTorch model to wts file](#convert-pytorch-model-to-wts-file)
* [Convert wts file to TensorRT model](#convert-wts-file-to-tensorrt-model)
* [Compile nvdsinfer_custom_impl_Yolo](#compile-nvdsinfer_custom_impl_yolo)
* [Testing model](#testing-model)

##

### Requirements
* Python3
```
sudo apt-get install python3 python3-dev python3-pip
pip3 install --upgrade pip
```

* OpenCV Python
```
sudo apt-get install libopencv-dev
pip3 install opencv-python
```

* Matplotlib
```
pip3 install matplotlib
```

* Matplotlib (for Jetson plataform)
```
sudo apt-get install python3-matplotlib
```

* Scipy
```
pip3 install scipy
```

* tqdm
```
pip3 install tqdm
```

* Pandas
```
pip3 install pandas
```

* seaborn
```
pip3 install seaborn
```

* PyTorch
```
pip3 install torch torchvision
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
git clone https://github.com/DanaHan/Yolov5-in-Deepstream-5.0.git yolov5converter
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone https://github.com/ultralytics/yolov5.git
```

Note: checkout TensorRTX repo to 3.0/3.1 YOLOv5 version
```
cd tensorrtx
git checkout '6d0f5cb'
```

2. Download latest YoloV5 (YOLOv5s, YOLOv5m, YOLOv5l or YOLOv5x) weights to yolov5/weights directory (example for YOLOv5s)
```
wget https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt -P yolov5/weights/
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

<br />

Note: if you want to generate wts file to another YOLOv5 model (YOLOv5m, YOLOv5l or YOLOv5x), edit get_wts.py file changing yolov5s to your model name
```
model = torch.load('weights/yolov5s.pt', map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

f = open('yolov5s.wts', 'w')
```

##

### Convert wts file to TensorRT model
1. Replace yololayer files from tensorrtx/yolov5 folder to yololayer and hardswish files from yolov5converter
```
mv yolov5converter/yololayer.cu tensorrtx/yolov5/yololayer.cu
mv yolov5converter/yololayer.h tensorrtx/yolov5/yololayer.h
```

2. Move generated yolov5s.wts file to tensorrtx/yolov5 folder (example for YOLOv5s)
```
cp yolov5/yolov5s.wts tensorrtx/yolov5/yolov5s.wts
```

3. Build tensorrtx/yolov5
```
cd tensorrtx/yolov5
mkdir build
cd build
cmake ..
make
```

4. Convert to TensorRT model (yolov5s.engine file will be generated in tensorrtx/yolov5/build folder)
```
sudo ./yolov5 -s
```

5. Create a custom yolo folder and copy generated files (example for YOLOv5s)
```
mkdir /opt/nvidia/deepstream/deepstream-5.0/sources/yolo
cp yolov5s.engine /opt/nvidia/deepstream/deepstream-5.0/sources/yolo/yolov5s.engine
```

<br />

Note: by default, yolov5 script generate model with batch size = 1, FP16 mode and s model.
```
#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define NET s  // s m l x
```
Edit yolov5.cpp file before compile if you want to change this parameters.

##

### Compile nvdsinfer_custom_impl_Yolo
1. Run command
```
sudo chmod -R 777 /opt/nvidia/deepstream/deepstream-5.0/sources/
```

2. Donwload [my external/yolov5 folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/external/yolov5) and move files to created yolo folder

3. Compile lib
```
cd /opt/nvidia/deepstream/deepstream-5.0/sources/yolo
CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
```

##

### Testing model
Use my edited [deepstream_app_config.txt](https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/master/external/yolov5/deepstream_app_config.txt) and [config_infer_primary.txt](https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/master/external/yolov5/config_infer_primary.txt) files available in [my external/yolov5 folder](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master/external/yolov5)

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
