# YOLOv7 usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yoloV7 file](#edit-the-config_infer_primary_yolov7-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOv7 repo and install the requirements

```
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip3 install -r requirements.txt
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `gen_wts_yoloV7.py` file from `DeepStream-Yolo/utils` directory to the `yolov7` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) releases (example for YOLOv7)

```
wget hhttps://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

**NOTE**: You can use your custom model, but it is important to keep the YOLO model reference (`yolov7_`) in you `cfg` and `weights`/`wts` filenames to generate the engine correctly.

#### 4. Reparameterize your model

[YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) checkpoints can't be directly converted in some cases and `gen_wts_yoloV7.py` script gives an eror such as model not supported. To solve this, you can first reparametrize yolov7 checkpoints (weights) to generate the engine file. 

Copy the `reparametrize.py` file from `DeepStream-Yolo/utils/reparameterize-yolov7` directory to the `yolov7` folder and then run.

```
python3 reparametrize.py --weights yolov7.pt --classes 80 
```

change number of classes and weights if you trained on custom dataset. Also, you can specify model config path and the path to a directory where you want to save your reparameterized checkpoints.

Now you can make engine file in the next step using your reparameterized checkpoints.

#### 5. Convert model

Generate the `cfg` and `wts` files (example for YOLOv7)

```
python3 gen_wts_yoloV7.py -w yolov7.pt
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

#### 6. Copy generated files

Copy the generated `cfg` and `wts` files to the `DeepStream-Yolo` folder.

##

### Compile the lib

Open the `DeepStream-Yolo` folder and compile the lib

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

##

### Edit the config_infer_primary_yoloV7 file

Edit the `config_infer_primary_yoloV7.txt` file according to your model (example for YOLOv7)

```
[property]
...
custom-network-config=yolov7.cfg
model-file=yolov7.wts
...
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yoloV7.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```
