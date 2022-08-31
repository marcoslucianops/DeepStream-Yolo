# YOLOv4-large usage

**NOTE**: Follow this guide for pytorch YOLO large models from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4):
- YOLOv4-P5
- YOLOv4-P6
- YOLOv4-P7

**NOTE**: The yaml file is not required.

- [Convert model](#convert-model)
- [Compile the lib](#compile-the-lib)
- [Edit the config_infer_primary_yoloV4-large file](#edit-the-config_infer_primary_yolov4-large-file)
- [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
- [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOv4 repo and install the requirements

```
git clone https://github.com/WongKinYiu/ScaledYOLOv4
cd ScaledYOLOv4
git checkout yolov4-large
```

**NOTE**: It is recommended to use a docker container:
>```
>[OPTIONAL, run from ScaledYOLOv4]
>nvidia-docker run --name yolov4_csp_nvds --rm -it -v $PWD:/ScaledYOLOv4 --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3
>cd /ScaledYOLOv4
>```

Install mish cuda. If using docker, install outside the shared volume
```
cd ~/
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python3 setup.py build install
```


#### 2. Copy conversor

Copy the `gen_wts_yoloV4-large.py` file from `DeepStream-Yolo/utils` directory to the `ScaledYOLOv4` folder.

#### 3. Download the model

Download the `pt` file from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4). See `yolov4-large` branch in the benchmarking tables.


**NOTE**: You can use your custom model, but it is important to keep the YOLO model reference (`yolov4_`) in you `cfg` and `weights`/`wts` filenames to generate the engine correctly.

#### 4. Convert model

Generate the `cfg` and `wts` files (example for YOLOv5s)

```
python3 gen_wts_yoloV4-large.py -w yolov4-P6.pt
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Examples for 1280

```
-s 1280
-s 1280 1280
```

#### 5. Copy generated files

Copy the generated `cfg` and `wts` files to the `DeepStream-Yolo` folder.

##

### Compile the lib

Open the `DeepStream-Yolo` folder and compile the lib

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
  ```

##

### Edit the config_infer_primary_yoloV4-large file

Edit the `config_infer_primary_yoloV4-large.txt` file according to your model (example for YOLOv4-P6)

```
[property]
...
custom-network-config=yolov4-p6.cfg
model-file=yolov4-p6.wts
...
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yoloV4-large.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```
