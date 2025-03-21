# Instructions for building GPU enabled tensorflow

Note, this README used the [requirements file in this repository](./requirements.txt). It originates from a `requirements.txt` developed by Mason Hurley for running on GPU nodes at the Minnesota Supercomputing Institute. Your milage may vary, depending on your own computing needs and resources

## IMPORTANT

you must do this installation on an MSI node with GPU present. There are possibly some hacks/workaround to do it other ways, but I have not trialed any of them. you must request 4-5 hours on your interatcive GPU node to run this. it is a slow execution.

## Install

   1. Get a GPU node. Here are my suggestions for what to request:

```bash
 clear; date ; srun -N 1 -n1  -c 1  --tmp=200gb --mem=48gb -t 270 -p interactive-gpu --gres=gpu:a40:1 --account=iaa --pty bash
```

   2. Load the new MSI conda packages

```bash
module load miniforge
```

   3. Set your pkgs_dirs in conda config. Above it requests a `tmp` dir of 200gb, so best to use it for storing packages on the node

```bash

conda config --add pkgs_dirs /tmp # or /scratch.local either is fine
```

   4. Build the environment. In this directory is the [`requirements.txt`](./requirements.txt) used in this step. Download or create that file wherever you inted to build the enviroment from.

```bash
wget https://github.umn.edu/uspatial/StLouisRiver_NERR/unet/requirements.txt
conda create --name tensorflow_gpu --file requirements.txt -c conda-forge
```

   5. Activate  the environment, and configure the jupyter environment

 ```bash
 source activate tensorflow_gpu
 python -m ipykernel install --user --name mason_tf_gpu --display-name "Python (mason_tf_gpu)" # or change display-name to suite you
 ```

   6. On a GPU node, ensure that a device is found. In this step you will likely see lots of warnings. Those are not a problem:

 ```bash
 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

 ...

 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
 ```

   7. Install [segmentation-models](https://github.com/qubvel/segmentation_models?tab=readme-ov-file#installation). `Requirements.txt`has the keras-applications needed included. but the last two packages will be needed to installed by pip.

```bash
pip install image-classifiers==1.0.*
pip install efficientnet==1.0.*
```

   8. Once that is done, nearly ready. Now need to edit the files in `efficientnet` to correct a problem. Follow these [instructions by DigitalSreeni to do that](https://www.youtube.com/watch?v=syJZxDtLujs).
