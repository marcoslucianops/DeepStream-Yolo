# YOLOR usage

**NOTE**: You need to use the main branch of the YOLOR repo to convert the model.

**NOTE**: The cfg file is required.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolor file](#edit-the-config_infer_primary_yolor-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOR repo and install the requirements

```
git clone https://github.com/WongKinYiu/yolor.git
cd yolor
pip3 install -r requirements.txt
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `gen_wts_yolor.py` file from `DeepStream-Yolo/utils` directory to the `yolor` folder.

#### 3. Download the model

Download the `pt` file from [YOLOR](https://github.com/WongKinYiu/yolor) repo.

**NOTE**: You can use your custom model, but it is important to keep the YOLO model reference (`yolor_`) in you `cfg` and `weights`/`wts` filenames to generate the engine correctly.

#### 4. Convert model

Generate the `cfg` and `wts` files (example for YOLOR-CSP)

```
python3 gen_wts_yolor.py -w yolor_csp.pt -c cfg/yolor_csp.cfg
```

#### 5. Copy generated files

Copy the generated `cfg` and `wts` files to the `DeepStream-Yolo` folder

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

### Edit the config_infer_primary_yolor file

Edit the `config_infer_primary_yolor.txt` file according to your model (example for YOLOR-CSP)

```
[property]
...
custom-network-config=yolor_csp.cfg
model-file=yolor_csp.wts
...
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yolor.txt
```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```
