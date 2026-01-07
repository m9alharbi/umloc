# UMLoc: Uncertainty-Aware Map-Constrained Inertial Localization with Quantified Bounds

UMLoc is a deep learning framework for 2D pedestrian inertial localization that combines IMU-based uncertainty estimation with map-aware trajectory generation.

## Overview
The method consists of two main components:
1. **LSTM-based Quantile Regression Module** that predicts lower and upper bounds on velocity uncertainty from IMU data.
2. **Conditional Generative Adversarial Network (CGAN)** that uses a distance map, quantile bounds and cross attention to generate physically feasible trajectories.

This design improves robustness, achieves drift-resilience, and enforces map compliance without visual sensing.

## Key Features
- IMU-only localization (no vision required)
- Uncertainty-aware prediction via quantile regression
- Map-aware trajectory generation
- Robust to noise and long-term drift
- End-to-end trainable framework


```shell
git clone https://github.com/USERNAME/umloc.git
cd umloc
./create-conda-env.sh
```

## Our Dataset
Experiments are conducted using:

- Smartphone IMU data
- Ground-truth trajectories from a stereo camera system
- 2D distance maps generated from the environment

Dataset details will be released upon publication.

The Android application and ZED2i code for data collection along with processing codes will be available upon publication.
## Pre-trained model

Pretrained model details will be released upon publication.

## Training and testing
### running
`./run_train.sh`

`./run_test.sh`

### Configuration
`--dataset_directory`: path of dataset  
`--output_directory`: path for results  
`--model_type`: which part of the model to be trained Quantile model, CGAN or end-to-end  
`train`: training mode  
`--train_list`: path of training dataset list  
`--val_list`: path of validation dataset list  
`test`: testing mode  
`--test_list`: path of testing dataset list  
`--lstm_path`: path of quantile model  
`--gan_path`: path of CGAN model  

## Acknowledgment
The authors would like to acknowledge the assistance from the KAUST Facilities and KAUST Supercomputing Laboratory (KSL) for making the training possible.

