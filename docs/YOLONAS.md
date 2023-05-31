# YOLO-NAS usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolonas file](#edit-the-config_infer_primary_yolonas-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLO-NAS repo and install the requirements

```
git clone https://github.com/Deci-AI/super-gradients.git
cd super-gradients
pip3 install -r requirements.txt
python3 setup.py install
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolonas.py` file from `DeepStream-Yolo/utils` directory to the `super-gradients` folder.

#### 3. Download the model

Download the `pth` file from [YOLO-NAS](https://sghub.deci.ai/) releases (example for YOLO-NAS S)

```
wget https://sghub.deci.ai/models/yolo_nas_s_coco.pth
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO-NAS S)

```
python3 export_yolonas.py -m yolo_nas_s -w yolo_nas_s_coco.pth --simplify --dynamic
```

**NOTE**: If you are using DeepStream 5.1, use opset 12 or lower. The default opset is 14.

```
--opset 12
```

**NOTE**: Model names

```
-m yolo_nas_s
```

or

```
-m yolo_nas_m
```

or

```
-m yolo_nas_l
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

#### 5. Copy generated file

Copy the generated ONNX model file to the `DeepStream-Yolo` folder.

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
  CUDA_VER=11.1 LEGACY=1 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 5.1 on Jetson platform

  ```
  CUDA_VER=10.2 LEGACY=1 make -C nvdsinfer_custom_impl_Yolo
  ```

##

### Edit the config_infer_primary_yolonas file

Edit the `config_infer_primary_yolonas.txt` file according to your model (example for YOLO-NAS S with 80 classes)

```
[property]
...
onnx-file=yolo_nas_s_coco.onnx
model-engine-file=yolo_nas_s_coco.onnx_b1_gpu0_fp32.engine
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYoloE
...
```

**NOTE**: The **YOLO-NAS** resizes the input with left/top padding. To get better accuracy, use

```
maintain-aspect-ratio=1
symmetric-padding=0
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yolonas.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
