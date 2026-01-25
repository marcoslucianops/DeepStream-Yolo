# YOLO-World usage

YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions. Unlike traditional YOLO models, YOLO-World requires pre-setting the classes to detect, which generates CLIP text embeddings that are "baked" into the exported model.

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yoloworld file](#edit-the-config_infer_primary_yoloworld-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)
* [Custom classes](#custom-classes)

##

### Convert model

#### 1. Download the Ultralytics repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloworld.py` file from `DeepStream-Yolo/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv8-World](https://github.com/ultralytics/assets/releases/) releases (example for YOLOv8s-worldv2)

```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOv8s-worldv2 with COCO 80 classes)

```
python3 export_yoloworld.py -w yolov8s-worldv2.pt --dynamic
```

**NOTE**: By default, the export script uses COCO 80 classes. See [Custom classes](#custom-classes) section for using custom class names.

**NOTE**: To change the inference size (default: 640)

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

Copy the generated ONNX model file and labels.txt file to the `DeepStream-Yolo` folder.

```
cp yolov8s-worldv2.onnx labels.txt /path/to/DeepStream-Yolo/
```

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

### Edit the config_infer_primary_yoloworld file

Edit the `config_infer_primary_yoloworld.txt` file according to your model (example for YOLOv8s-worldv2 with 80 classes)

```
[property]
...
onnx-file=yolov8s-worldv2.onnx
...
num-detected-classes=80
...
labelfile-path=labels.txt
...
parse-bbox-func-name=NvDsInferParseYolo
...
```

**NOTE**: The **YOLO-World** resizes the input with center padding. To get better accuracy, use

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
config-file=config_infer_primary_yoloworld.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.

##

### Custom classes

YOLO-World supports open-vocabulary detection, meaning you can define any classes you want to detect. Use the `--custom-classes` parameter to specify your custom classes:

```
python3 export_yoloworld.py -w yolov8s-worldv2.pt --custom-classes "person, car, dog" --dynamic --simplify
```

**NOTE**: Classes should be comma-separated. Spaces around class names are automatically trimmed.

More examples:

```
# Single class
python3 export_yoloworld.py -w yolov8s-worldv2.pt --custom-classes "person" --dynamic

# Multiple classes
python3 export_yoloworld.py -w yolov8s-worldv2.pt --custom-classes "person, car, truck, bus, motorcycle" --dynamic

# Default COCO 80 classes (no --custom-classes)
python3 export_yoloworld.py -w yolov8s-worldv2.pt --dynamic
```

**IMPORTANT**: After changing the classes, you must:

1. Re-export the ONNX model with the new `--custom-classes` parameter
2. Update `num-detected-classes` in the config file to match the number of classes
3. Delete the old TensorRT engine file (if exists) so a new one will be generated
4. The generated `labels.txt` file will automatically contain your custom classes

Example for 3 custom classes:

```
[property]
...
num-detected-classes=3
...
```
