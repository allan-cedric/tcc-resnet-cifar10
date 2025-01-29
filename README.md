## Performance of two popular embedded systems using Residual Neural Networks for Image Classification

### Warning

This README are under development

### Abstract

Today, many neural networks generally rely on extensive computational power for
training and execution, typically being confined to large servers. This traditional approach
to handling neural networks can have significant impacts related to execution latency, data
privacy, energy consumption, and financial costs. For this reason, the field of TinyML aims
to bring the execution of neural networks to embedded systems, which are generally more
cost-effective and energy-efficient. However, there are several challenges to deploying neural
networks on resource-constrained systems, often requiring the application of neural network
compression techniques to reduce model size. With this in mind, this work aims to evaluate the
performance and energy efficiency of two popular embedded systems, ESP32 and Raspberry
Pi 3B+, in the task of classifying images from the CIFAR-10 dataset using Residual Neural
Networks (ResNets). In this work, the Raspberry Pi platform achieved the best accuracy result
(ResNet-20: 0,8659), the best inference time result (ResNet-8: 0,009 s), and the best energy
consumption per inference result (ResNet-8: 0,024 J). Meanwhile, the ESP32 achieved the
best power consumption result (ResNet-8: 0,38 W). The code for this work and the results are
available at: https://github.com/allan-cedric/tcc-resnet-cifar10.

### Link
The complete thesis are available [here](./TCC_2024.pdf) for more information.

### About this repository

#### Requisites

* PlatformIO (I recommend to use the VSCode Extension)
* Python 3.10
* An ESP32 board or Raspberry Pi board (In this work is used EPS32-DevKit-V1 and Raspberry Pi 3B+)

#### Models

`models/resnet{N}/`: Directory of a ResNet-N model. Includes the original model (`.keras`), the optimized model (`.tflite`) and the training log of the original model (`.csv`).

#### Results

#### Running on ESP32

#### Running on Raspberry Pi
