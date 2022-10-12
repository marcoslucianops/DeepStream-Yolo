# YOLOv5 usage

**NOTE**: You can use the main branch of the YOLOv5 repo to convert all model versions.

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yoloV5 file](#edit-the-config_infer_primary_yolov5-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOv5 repo and install the requirements

```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip3 install -r requirements.txt
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `gen_wts_yoloV5.py` file from `DeepStream-Yolo/utils` directory to the `yolov5` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv5](https://github.com/ultralytics/yolov5/releases/) releases (example for YOLOv5s 6.1)

```
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
```

**NOTE**: You can use your custom model, but it is important to keep the YOLO model reference (`yolov5_`) in you `cfg` and `weights`/`wts` filenames to generate the engine correctly.

#### 4. Convert model

Generate the `cfg` and `wts` files (example for YOLOv5s)

```
python3 gen_wts_yoloV5.py -w yolov5s.pt
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

#### 5. Copy generated files

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

### Edit the config_infer_primary_yoloV5 file

Edit the `config_infer_primary_yoloV5.txt` file according to your model (example for YOLOv5s)

```
[property]
...
custom-network-config=yolov5s.cfg
model-file=yolov5s.wts
...
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yoloV5.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```
