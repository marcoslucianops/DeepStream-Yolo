# YOLOv10 usage

**NOTE**: The yaml file is not required.

- [YOLOv10 usage](#yolov10-usage)
  - [](#)
    - [Convert model](#convert-model)
      - [1. Download the YOLOv10 repo and install the requirements](#1-download-the-yolov10-repo-and-install-the-requirements)
      - [2. Copy conversor](#2-copy-conversor)
      - [3. Download the model](#3-download-the-model)
      - [4. Convert model](#4-convert-model)
      - [5. Copy generated files](#5-copy-generated-files)
  - [](#-1)
    - [Compile the lib](#compile-the-lib)
  - [](#-2)
    - [Edit the config\_infer\_primary\_yolov10 file](#edit-the-config_infer_primary_yolov10-file)
  - [](#-3)
    - [Edit the deepstream\_app\_config file](#edit-the-deepstream_app_config-file)
  - [](#-4)
    - [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOv10 repo and install the requirements

```
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
pip3 install -r requirements.txt
pip install -e .
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolov10.py` file from `DeepStream-Yolo/utils` directory to the `yolov10` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv10](https://github.com/THU-MIG/yolov10/releases) releases (example for YOLOv10s)

```
wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOv10s)

```
python3 export_yoloV10.py -w yolov10s.pt --dynamic
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

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```


**NOTE**: To change maximum number of Detections (example for max_det = 300 )

```
--max_det 300
```

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 16.

```
--opset 12
```

#### 5. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo` folder and compile the lib

2. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  DeepStream 5.1 = 11.1
  ```

* Jetson platform

  ```
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
  ```

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

##

### Edit the config_infer_primary_yolov10 file

Edit the `config_infer_primary_yolov10.txt` file according to your model (example for YOLOv10s with 80 classes)

```
[property]
...
onnx-file=yolov10s.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYoloE
...
```

**NOTE**: The **YOLOv10** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yolov10.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
