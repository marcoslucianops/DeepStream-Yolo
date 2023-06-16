# PP-YOLOE / PP-YOLOE+ usage

**NOTE**: You can use the release/2.6 branch of the PPYOLOE repo to convert all model versions.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_ppyoloe_plus file](#edit-the-config_infer_primary_ppyoloe_plus-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the PaddleDetection repo and install the requirements

https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/docs/tutorials/INSTALL.md

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_ppyoloe.py` file from `DeepStream-Yolo/utils` directory to the `PaddleDetection` folder.

#### 3. Download the model

Download the `pdparams` file from [PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/ppyoloe) releases (example for PP-YOLOE+_s)

```
wget https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for PP-YOLOE+_s)

```
pip3 install onnx onnxsim onnxruntime
python3 export_ppyoloe.py -w ppyoloe_plus_crn_s_80e_coco.pdparams -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml --dynamic
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

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 11.

```
--opset 12
```

#### 5. Copy generated files

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

### Edit the config_infer_primary_ppyoloe_plus file

Edit the `config_infer_primary_ppyoloe_plus.txt` file according to your model (example for PP-YOLOE+_s with 80 classes)

```
[property]
...
onnx-file=ppyoloe_plus_crn_s_80e_coco.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYoloE
...
```

**NOTE**: If you are using the **legacy** model, you should edit the `config_infer_primary_ppyoloe.txt` file.

**NOTE**: The **PP-YOLOE+ and PP-YOLOE legacy** do not resize the input with padding. To get better accuracy, use

```
...
maintain-aspect-ratio=0
...
```

**NOTE**: The **PP-YOLOE+** uses zero mean normalization on the image preprocess. It is important to change the `net-scale-factor` according to the trained values.

```
...
net-scale-factor=0.0039215697906911373
...
```

**NOTE**: The **PP-YOLOE legacy** uses normalization on the image preprocess. It is important to change the `net-scale-factor` and `offsets` according to the trained values.

Default: `mean = 0.485, 0.456, 0.406` and `std = 0.229, 0.224, 0.225`

```
...
net-scale-factor=0.0173520735727919486
offsets=123.675;116.28;103.53
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
config-file=config_infer_primary_ppyoloe_plus.txt
```

**NOTE**: If you are using the **legacy** model, you should edit it to `config_infer_primary_ppyoloe.txt`.

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
