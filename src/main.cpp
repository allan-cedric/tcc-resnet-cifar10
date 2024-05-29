/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Arduino.h>
#include "esp_system.h"
#include <driver/uart.h>
#include <driver/gpio.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "constants.h"
#include "output_handler.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// area de memoria utilizada para entrada, saida e buffer intermediarios
// achar o valor otimo eh um multiplo de 16 e que seja o menor possivel (tentativa e erro)
// constexpr int kTensorArenaSize = 3422 * 16; // ResNet8
constexpr int kTensorArenaSize = 4647 * 16; // ResNet14
// constexpr int kTensorArenaSize = 4847 * 16; // ResNet20
// constexpr int kTensorArenaSize = 5048 * 16; // ResNet26
// constexpr int kTensorArenaSize = 5248 * 16; // ResNet32
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

#define INPUT_IMAGE_WIDTH 32
#define INPUT_IMAGE_HEIGHT 32
#define INPUT_IMAGE_CHANNELS 3
int totalExpectedDataAmount = INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT * INPUT_IMAGE_CHANNELS;

#define UART_NUMBER (UART_NUM_0)
#define TXD_PIN (GPIO_NUM_1) //(U0TXD)
#define RXD_PIN (GPIO_NUM_3) //(U0RXD)
#define RX_BUF_SIZE 1024

void initUart(uart_port_t uart_num)
{
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB};

    // We will not use a buffer for sending data.
    uart_driver_install(uart_num, RX_BUF_SIZE * 2, 0, 0, NULL, 0);

    // Configure UART parameters
    ESP_ERROR_CHECK(uart_param_config(uart_num, &uart_config));
    // Cnfigure the physical GPIO pins to which the UART device will be connected.
    ESP_ERROR_CHECK(uart_set_pin(uart_num, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

void readUartBytes(float *data, int imageSize)
{
    uint8_t *rxBuffer = (uint8_t *)malloc(RX_BUF_SIZE + 1);
    int rxIdx = 0;
    int rxBytes = 0;

    for (;;)
    {
        rxBytes = uart_read_bytes(UART_NUMBER, rxBuffer, RX_BUF_SIZE, 1000 / portTICK_PERIOD_MS);
        if(rxBytes < 0)
          esp_restart();
        if (rxBytes > 0)
        {
            for (int i = 0; i < rxBytes; rxIdx++, i++)
            {
                data[rxIdx] = static_cast<float>(rxBuffer[i]) / 255.0f;
            }
        }
        if (rxIdx >= imageSize - 1)
        {
            rxIdx = 0;
            break;
        }
    }
    return;
}

int sendData(const char *data)
{
    const int len = strlen(data);
    const int txBytes = uart_write_bytes(UART_NUMBER, data, len);
    return txBytes;
}

void normalizeImageData(float *data, int imageSize)
{
    for (int i = 0; i < imageSize; i++)
    {
        data[i] = data[i] / 255.0f;
    }
}

void sendBackPredictions(TfLiteTensor *output, double total_time, int ret)
{
    // Read the predicted y values from the model's output tensor
    char str[500] = {0};
    char buf[20] = {0};
    int numElements = output->dims->data[1];
    for (int i = 0; i < numElements; i++)
    {
        sprintf(buf, "%e,", static_cast<float>(output->data.f[i]));
        strcat(str, buf);
    }

    double total_time_ms = total_time / 1000.0;
    sprintf(buf, "%e,", total_time_ms);
    strcat(str, buf);

    sprintf(buf, "%i,", ret);
    strcat(str, buf);

    strcat(str, "\n");
    sendData(str);
}

int doInference()
{
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        // Serial.println("erro");
        digitalWrite(LED_BUILTIN, LOW);
        return invoke_status;
    }
    digitalWrite(LED_BUILTIN, HIGH);
    return 0;
}

// The name of this function is important for Arduino compatibility.
void setup() {

  initUart(UART_NUMBER);

  pinMode(LED_BUILTIN, OUTPUT);

  // ESP32 Serial
  // Serial.begin(115200);
  // delay(2000);

  // Serial.print(getCpuFrequencyMhz()); // 240 MHz
  // while(1);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  
  // ResNet8
  // model = tflite::GetModel(models_resnet8_resnet8_model_optimized_tflite);

  // ResNet14
  model = tflite::GetModel(models_resnet14_resnet14_model_optimized_tflite);

  // ResNet20
  // model = tflite::GetModel(models_resnet20_resnet20_model_optimized_tflite);

  // ResNet26
  // model = tflite::GetModel(models_resnet26_resnet26_model_optimized_tflite);

  // ResNet32
  // model = tflite::GetModel(models_resnet32_resnet32_model_optimized_tflite);
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<9> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (resolver.AddAveragePool2D() != kTfLiteOk) {
    return;
  }
  if (resolver.AddRelu() != kTfLiteOk) {
    return;
  }
  if (resolver.AddAdd() != kTfLiteOk) {
    return;
  }
  if (resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (resolver.AddQuantize() != kTfLiteOk) {
    return;
  }
  if (resolver.AddDequantize() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Serial.print("arena_used_bytes: ");
  // Serial.println(interpreter->arena_used_bytes());
  // while(1);

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  // pronto para iniciar
  digitalWrite(LED_BUILTIN, HIGH);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  readUartBytes(input->data.f, totalExpectedDataAmount);
  int64_t start_time = esp_timer_get_time();
  int ret = doInference();
  int64_t total_time = (esp_timer_get_time() - start_time);
  // Serial.println(total_time/1000);
  sendBackPredictions(output, (double)total_time, ret);
}