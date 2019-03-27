# Sushi Detection model with Tensorflow
This is Just example for demo. <br>
Last updated: 3/28/2019 with TensorFlow v1.12 <br>
## Before you start.... <br>
### Setting Up DLVM on Azure (Ubuntu)
Please following [the Quickstarts.](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-deep-learning-dsvm)
### Tensorflow Object Detection
Basically following [the installation.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

#### 1. Clone the repo - Tensorflow models
https://github.com/tensorflow/models.git <br>
Any SSH and telnet clients are fine! <br>

#### 2. Install Python, tensorflow-gpu, cuDNN, CUDA - specific version
Here is [the version list.](https://www.tensorflow.org/install/source#tested_build_configurations)

`$python --version` <br>
Should be "Python 3.5.5 :: Anaconda custom (64-bit)" pre-installed on DLVM. <br>

You can also check nvidia driver version. <br>
`$cat /proc/driver/nvidia/version` <br>


`$pip install tensorflow-gpu==1.12.0` <br>
`$cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2` <br>
Will show like "#define CUDNN_MAJOR 7 define CUDNN_MINOR 2 define CUDNN_PATCHLEVEL 1" cuDNN version 7.0 should be pre-installed. <br>
`$>nvcc --version` <br>
Will show like "Cuda compilation tools, release 9.0, V9.0.176" CUDA version 9.0 should be pre-installed. <br>
`$nvidia-smi` <br>
Check GPU Card info. <br>

#### 3. Install libraries
`$pip install --user Cython` <br>
`$pip install --user contextlib2` <br>
`$pip install --user pillow` <br>
`$pip install --user lxml` <br>
`$pip install --user jupyter` <br>
`$pip install --user matplotlib` <br>

#### 4. Install COCO API
From models/research <br>
`$git clone https://github.com/cocodataset/cocoapi.git` <br>
`$cd cocoapi/PythonAPI` <br>
`$make` <br>
`$cp -r pycocotools <path_to_tensorflow>/models/research/` <br>

#### 5. Protobuf Compilation
From tensorflow/models/research/ <br>
`$wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip` <br>
`$unzip protobuf.zip` <br>

#### 6. Add Libraries to PYTHONPATH
From tensorflow/models/research/ <br>
`$export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim` <br>

#### 7. Testing the Installation
From tensorflow/models/research/ <br>
`$python object_detection/builders/model_builder_test.py` <br>
<br>
If showing "OK" that means you successfully completed the installation! <br>
<br>
#### 8. Access to Jupyter Notebook
`$jupyter notebook --ip=0.0.0.0 --port=8888` <br>
Copy the URL like "http://*****:8888/?token=**************" <br>

Then go to Azure portal and add inbound port rule "8888" <br>

Go to browser and paste "http://(DNS name - copy from Azure Portal):8888/?token=***********" <br>

**Now ready to code with tensorflow!**

----------------------------------------------------
## A. Annotate your own dataset <br>
### Annotate dataset <br>
[VoTT 1.5](https://github.com/Microsoft/VoTT/releases/download/1.5.0/VoTT-win32-x64.zip) <br>
### Export to Tensorflow Pascal VOC format <br>
You will get the four below. <br>
・Annotations folder <br>
・ImageSets folder <br>
・JPEGImages folder <br>
・pascal_label_map.pbtxt file <br>
## B. Train and Test data preparation <br>
### Convert XML to CSV <br>
`$python xml_to_csv.py`<br>
You need to split the csv file into two: train.csv and test.csv before the next step. <br>
### Generate tf-record file (Train) <br>
`$python generate_tfrecord.py --csv_input=train_labels.csv  --output_path=train.record --label_map_path=pascal_label_map.pbtxt`<br>
### Generate tf-record file (Test) <br>
`$python generate_tfrecord.py --csv_input=test_labels.csv  --output_path=test.record --label_map_path=pascal_label_map.pbtxt`<br>
<br>
## C. Train and Evaluate model <br>
### Train model <br>
`$python train.py --logtostderr --pipeline_config_path=sushi_resnet_101.config --train_dir=checkpoints_sushi`<br>
### Evaluate model <br>
`$python eval.py --logtostderr --pipeline_config_path=sushi_resnet_101.config --checkpoint_dir=checkpoints_sushi --eval_dir=checkpoint_sushi`<br>
### Tensorboard <br>
`$tensorboard --logdir=checkpoints_sushi --port 6008` <br>
