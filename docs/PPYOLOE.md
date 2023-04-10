# PP-YOLOE / PP-YOLOE+ usage

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_ppyoloe_plus file](#edit-the-config_infer_primary_ppyoloe_plus-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the PaddleDetection repo and install the requirements

https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/INSTALL.md

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `gen_wts_ppyoloe.py` file from `DeepStream-Yolo/utils` directory to the `PaddleDetection` folder.

#### 3. Download the model

Download the `pdparams` file from [PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/ppyoloe) releases (example for PP-YOLOE+_s)

```
wget https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams
```

**NOTE**: You can use your custom model, but it is important to keep the YOLO model reference (`ppyoloe_`) in you `cfg` and `weights`/`wts` filenames to generate the engine correctly.

#### 4. Convert model

Generate the `cfg` and `wts` files (example for PP-YOLOE+_s)

```
python3 gen_wts_ppyoloe.py -w ppyoloe_plus_crn_s_80e_coco.pdparams -c configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml
```

#### 5. Copy generated files

Copy the generated `cfg` and `wts` files to the `DeepStream-Yolo` folder.

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

### Edit the config_infer_primary_ppyoloe_plus file

Edit the `config_infer_primary_ppyoloe_plus.txt` file according to your model (example for PP-YOLOE+_s with 80 classes)

```
[property]
...
custom-network-config=ppyoloe_plus_crn_s_80e_coco.cfg
model-file=ppyoloe_plus_crn_s_80e_coco.wts
...
num-detected-classes=80
...
```

**NOTE**: If you use the **legacy** model, you should edit the `config_infer_primary_ppyoloe.txt` file.

**NOTE**: The **PP-YOLOE+** uses zero mean normalization on the image preprocess. It is important to change the `net-scale-factor` according to the trained values.

```
net-scale-factor=0.0039215697906911373
```

**NOTE**: The **PP-YOLOE (legacy)** uses normalization on the image preprocess. It is important to change the `net-scale-factor` and `offsets` according to the trained values.

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
config-file=config_infer_primary_ppyoloe_plus.txt
```

**NOTE**: If you use the **legacy** model, you should edit it to `config_infer_primary_ppyoloe.txt`.

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
