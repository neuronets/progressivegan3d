# Progressive GAN 

An implementation of progressive growing of GANs, purely in TensorFlow 2.0.

The code currently supports both 2D and 3D image generation.

## Install required packages

`pip install -r requirements.txt`

## Dataset Preparation

```
python main.py prepare
    --dataset path/to/data
    --save_path path/to/save/tfrecords
    --dimensionality 2/3
```

## Run Training

```
python main.py train 
    --dataset path/to/tfrecord/file
    --run_id path/to/save 
    --dimensionality 2/3 
    --latent_size latent_size
    --kiters_per_resolution 10 
    --kiters_per_transition 10 
    --gpus '/gpu:0' '/gpu:1' '/gpu:2' '/gpu:3' 
```

## Inference

Coming soon .. 

## Sample Results

### 2D Sagittal Mid Slices

### 3D T1 MRI Scans