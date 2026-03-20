# YOLO-Master usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolomaster file](#edit-the-config_infer_primary_yolomaster-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLO-Master repo and install the requirements

```
git clone https://github.com/Tencent/YOLO-Master.git
cd YOLO-Master
pip3 install -e .
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolomaster.py` file from `DeepStream-Yolo/utils` directory to the `YOLO-Master` folder.

#### 3. Download the model

Download the `pt` file from [YOLO-Master](https://github.com/Tencent/YOLO-Master/releases/) releases (example for YOLO-Master-N)

```
wget https://github.com/Tencent/YOLO-Master/releases/download/v0.0.0/yolo-master-v0.1-n.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO-Master-N)

```
python3 export_yolomaster.py -w yolo-master-v0.1-n.pt --dynamic
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

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 17.

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
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
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
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
  ```

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

##

### Edit the config_infer_primary_yolomaster file

Edit the `config_infer_primary_yolomaster.txt` file according to your model (example for YOLO-Master-N with 80 classes)

```
[property]
...
onnx-file=yolo-master-v0.1-n.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYolo
...
```

**NOTE**: The **YOLO-Master** resizes the input with center padding. To get better accuracy, use

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
config-file=config_infer_primary_yolomaster.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
