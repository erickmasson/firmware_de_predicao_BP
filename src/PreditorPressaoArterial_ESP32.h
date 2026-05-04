/*
 * PreditorPressaoArterial_ESP32.h (Arquivo de Cabeçalho)
 *
 * Define a "interface" da nossa classe de IA. Diz ao 'main.ino'
 * quais funções e variáveis a classe possui.
 */

#ifndef PREDITORPRESSAOARTERIAL_ESP32_H
#define PREDITORPRESSAOARTERIAL_ESP32_H

#include <Arduino.h>
#ifdef DEFAULT
#undef DEFAULT
#endif
#include <vector>
#include <ArduinoEigenDense.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

class PreditorPressaoArterial_ESP32
{
private:
  // Variáveis de Estado
  bool inicializado; // true se o modelo carregou, false se falhou

  // Variáveis do TensorFlow Lite (TFLite)
  tflite::ErrorReporter *reporter_erro;
  const tflite::Model *modelo;
  tflite::MicroInterpreter *interpretador;
  TfLiteTensor *tensor_entrada;
  TfLiteTensor *tensor_saida;

  alignas(16) static uint8_t arena_tensor[10 * 1024]; // 10KB

  // Constantes do Modelo (do script Python)
  Eigen::RowVector3d media_features;
  Eigen::RowVector3d desvio_features;
  const int NUM_FEATURES = 3;
  const double FREQ_AMOSTRAGEM = 400.0;

  // Funções auxiliares
  std::vector<int> segmentacao_adaptativa_cpp(const std::vector<double> &sinal);
  std::vector<int> find_vales(const std::vector<double> &sinal, const std::vector<int> &picos);
  Eigen::RowVector3d extrair_3_features_cpp(const std::vector<double> &segmento, int inicio, int pico);
  std::vector<double> gaussian_smooth(const std::vector<double> &segmento);
  double trapz(const std::vector<double> &segmento);

public:
  // Funções públicas

  // Construtor
  PreditorPressaoArterial_ESP32();
  // Destrutor
  ~PreditorPressaoArterial_ESP32();

  bool IsInitialized();
  std::pair<float, float> preverPressao(const std::vector<double> &segmento_bruto_ppg);
};

#endif