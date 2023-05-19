# YOLOX usage

**NOTE**: You can use the main branch of the YOLOX repo to convert all model versions.

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolox file](#edit-the-config_infer_primary_yolox-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOX repo and install the requirements

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolox.py` file from `DeepStream-Yolo/utils` directory to the `YOLOX` folder.

#### 3. Download the model

Download the `pth` file from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/releases/) releases (example for YOLOX-s standard)

```
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOX-s standard)

```
python3 export_yolox.py -w yolox_s.pth -c exps/default/yolox_s.py --simplify
```

#### 5. Copy generated files

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

* DeepStream 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo
  ```

##

### Edit the config_infer_primary_yolox file

Edit the `config_infer_primary_yolox.txt` file according to your model (example for YOLOX-s standard with 80 classes)

```
[property]
...
onnx-file=yolox_s.onnx
model-engine-file=yolox_s.onnx_b1_gpu0_fp32.engine
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYolo
...
```

**NOTE**: If you use the **legacy** model, you should edit the `config_infer_primary_yolox_legacy.txt` file.

**NOTE**: The **YOLOX standard** uses no normalization on the image preprocess. It is important to change the `net-scale-factor` according to the trained values.

```
net-scale-factor=0
```

**NOTE**: The **YOLOX legacy** uses normalization on the image preprocess. It is important to change the `net-scale-factor` and `offsets` according to the trained values.

Default: `mean = 0.485, 0.456, 0.406` and `std = 0.229, 0.224, 0.225`

```
net-scale-factor=0.0173520735727919486
offsets=123.675;116.28;103.53
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yolox.txt
```

**NOTE**: If you use the **legacy** model, you should edit it to `config_infer_primary_yolox_legacy.txt`.

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
