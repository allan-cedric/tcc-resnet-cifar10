# Performance of two popular embedded systems using Residual Neural Networks for Image Classification

## Abstract

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

## Thesis

The complete thesis is available [here](./TCC_2024.pdf) for more information.

## Repository structure

```
├── platformio.ini                          # PlatformIO config for ESP32
├── requirements-python3.10-tf.txt          # Python 3.10 + TensorFlow 2.15
├── requirements-python3.10-tflite.txt      # Python 3.10 + TensorFlow Lite 2.16
├── requirements-python3.12-tflite.txt      # Python 3.12 + TensorFlow Lite 2.16
├── generate_model_array.sh                 # Convert .tflite to C array via xxd
│
├── models/
│   └── resnet{N}/
│       ├── resnet{N}_model.{epoch}-{loss}.keras         # Trained TF model
│       ├── resnet{N}-model-optimized.tflite              # Quantized TFLite model
│       └── training_log.csv                              # Training history
│
├── src/                                    # ESP32 firmware (PlatformIO)
│   ├── main.cpp                            # UART inference loop
│   ├── model.h / model.cc                  # Model declarations
│   ├── constants.h / constants.cc          # Config (inferences per cycle)
│   ├── output_handler.h / output_handler.cc
│   └── resnet{N}-model-optimized.tflite.cc # Model as C array
│
├── raspberrypi_cifar10_test_resnet_tf.py       # RPi inference with TF
├── raspberrypi_cifar10_test_resnet_tflite.py   # RPi inference with TFLite
├── esp32_cifar10_test_resnet{N}.ipynb          # ESP32 inference via UART
│
├── gen_accuracy_graphics.ipynb              # Accuracy/F1 bar charts
├── gen_time_graphics.ipynb                  # Inference time bar charts
├── gen_power_graphics.ipynb                 # Power consumption bar charts
├── gen_model_mem_used.ipynb                 # Model size comparison
├── gen_train_graphics.ipynb                 # Training loss curves
│
├── results/
│   ├── accuracy/  time/  power/  train/     # Generated charts
│   └── res-{timestamp}-{model}-{platform}/  # Per-run results
│       ├── results-{timestamp}.json         # 10k inference results
│       ├── creport-{timestamp}.json         # Classification report
│       ├── cmatrix-{timestamp}.png           # Confusion matrix
│       └── um34c_data_*.csv                 # Power measurement data
│
└── lib/tflite-micro/                        # TFLite Micro for ESP32
```

## Requirements

- **Python 3.10 or 3.12** with dependencies from the appropriate `requirements-*.txt`
- **PlatformIO** (VSCode extension recommended) for the ESP32 firmware
- **ESP32-DevKit-V1** or **Raspberry Pi 3B+** (or similar boards)
- **UM34C USB power meter** (optional, for power measurements)

## Models

This repository includes five ResNet variants trained on CIFAR-10:

| Model | Parameters | .keras size | .tflite size | Compression |
|-------|-----------|-------------|--------------|-------------|
| ResNet-8  | ~78K  | ~1.03 MB | ~0.09 MB | ~11x |
| ResNet-14 | ~175K | ~2.24 MB | ~0.19 MB | ~12x |
| ResNet-20 | ~272K | ~3.45 MB | ~0.30 MB | ~11x |
| ResNet-26 | ~369K | ~4.66 MB | ~0.40 MB | ~12x |
| ResNet-32 | ~466K | ~5.87 MB | ~0.50 MB | ~12x |

Each model directory under `models/resnet{N}/` contains:
- The trained model in `.keras` format
- The quantized TFLite model (post-training int8 quantization)
- The training log (loss and validation loss per epoch)

## How to use

### 1. Set up the environment

```bash
pip install -r requirements-python3.10-tf.txt      # For training / TF on RPi
pip install -r requirements-python3.10-tflite.txt   # For TFLite on RPi
```

### 2. Run inference on Raspberry Pi

**With full TensorFlow:**
```bash
python3 raspberrypi_cifar10_test_resnet_tf.py models/resnet8/resnet8_model.*.keras
```

**With TFLite (optimized):**
```bash
python3 raspberrypi_cifar10_test_resnet_tflite.py models/resnet8/resnet8-model-optimized.tflite
```

Both scripts load the CIFAR-10 test set (10,000 images), run inference on each image, measure per-image inference time, and save results (JSON, classification report, confusion matrix) to `results/res-{timestamp}-{model}-{platform}/`.

### 3. Run inference on ESP32

#### 3.1 Convert TFLite model to C array

```bash
./generate_model_array.sh models/resnet8/resnet8-model-optimized.tflite
```

This generates `src/resnet8-model-optimized.tflite.cc`.

#### 3.2 Configure firmware

In `src/main.cpp`:
- Uncomment the `GetModel()` call for the desired model
- Set the correct tensor arena size (ResNet-8: 3422, ResNet-14: 4647, ResNet-20: 4847, ResNet-26: 5048, ResNet-32: 5248)

The tensor arena size in the code is specified in 16-byte units. For example, for ResNet-8:
```cpp
constexpr int kTensorArenaSize = 3422 * 16;
```

#### 3.3 Build and flash

Open the project in VSCode with PlatformIO, then build and upload to the ESP32.

#### 3.4 Run inference via UART

Open the appropriate notebook `esp32_cifar10_test_resnet{N}.ipynb`, configure the `SERIAL_PORT_NAME` variable (e.g., `/dev/ttyUSB0`), and run all cells. The notebook sends each CIFAR-10 test image to the ESP32 over UART, receives the predictions, and saves the results.

### 4. Generate graphics

Run any of the `gen_*.ipynb` notebooks to regenerate the comparative charts from the collected results:

- `gen_train_graphics.ipynb` — Training and validation loss curves
- `gen_accuracy_graphics.ipynb` — Accuracy and F1-score comparison
- `gen_time_graphics.ipynb` — Inference time comparison
- `gen_power_graphics.ipynb` — Power consumption comparison
- `gen_model_mem_used.ipynb` — Model size comparison

## Results

All inference results are organized under `results/` in timestamped directories for each combination of hardware and model. Each directory contains per-image inference data, classification reports, confusion matrices, and power measurement data (when available).

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.