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
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloV7.py` file from `DeepStream-Yolo/utils` directory to the `yolov7` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) releases (example for YOLOv7)

```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

**NOTE**: You can use your custom model.

#### 4. Reparameterize your model

[YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) and its variants cannot be directly converted to engine file. Therefore, you will have to reparameterize your model using the code [here](https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb). Make sure to convert your custom checkpoints in yolov7 repository, and then save your reparmeterized checkpoints for conversion in the next step.

#### 5. Convert model

Generate the ONNX model file (example for YOLOv7)

```
python3 export_yoloV7.py -w yolov7.pt --dynamic
```

**NOTE**: To convert a P6 model

```
--p6
```

**NOTE**: To change the inference size (defaut: 640 / 1280 for `--p6` models)

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

**NOTE**: To use implicit batch-size (example for batch-size = 4)

```
--batch 4
```

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 12.

```
--opset 12
```

#### 6. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo` folder.

##

### Compile the lib

Open the `DeepStream-Yolo` folder and compile the lib

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

##

### Edit the config_infer_primary_yoloV7 file

Edit the `config_infer_primary_yoloV7.txt` file according to your model (example for YOLOv7 with 80 classes)

```
[property]
...
onnx-file=yolov7.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYolo
...
```

**NOTE**: The **YOLOv7** resizes the input with center padding. To get better accuracy, use

```
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```

**NOTE**: By default, the dynamic batch-size is set. To use implicit batch-size, uncomment the line

```
...
force-implicit-batch-dim=1
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

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
