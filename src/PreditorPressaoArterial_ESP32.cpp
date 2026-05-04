/*
 * PreditorPressaoArterial_ESP32.cpp (Arquivo de Implementação)
 *
 * Contém a lógica "pesada" da IA. Implementa as funções
 * declaradas no arquivo .h.
 */

#include "PreditorPressaoArterial_ESP32.h" // Inclui o cabeçalho
#include "model_data.h"                    // Onde o modelo (array 'ann_MLP_3F_tflite') está
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>

// Define o bloco de memória estático para a arena do TFLite
alignas(16) uint8_t PreditorPressaoArterial_ESP32::arena_tensor[10 * 1024];

// Construtor da classe (executado durante o setup())
PreditorPressaoArterial_ESP32::PreditorPressaoArterial_ESP32() : inicializado(false)
{
  // 1. Define as constantes de normalização (do script Python)
  media_features << 0.28579714, 0.20255669, 0.38727512;
  desvio_features << 0.0787428, 0.07602341, 0.12433794;

  // 2. Inicializa o TFLite
  static tflite::MicroErrorReporter micro_error_reporter;
  reporter_erro = &micro_error_reporter;

  // Carrega o modelo (array) do arquivo model_data.h
  modelo = tflite::GetModel(ann_MLP_3F_tflite);

  // 3. Define quais "operações" (Ops) o modelo usa
  static tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected(); // Camada densa
  resolver.AddRelu();           // Função de ativação ReLU

  // 4. Cria o Interpretador
  static tflite::MicroInterpreter static_interpreter(modelo, resolver, arena_tensor, sizeof(arena_tensor), reporter_erro);
  interpretador = &static_interpreter;

  // 5. Aloca a memória (arena) para os tensores
  if (interpretador->AllocateTensors() != kTfLiteOk)
  {
    reporter_erro->Report("AllocateTensors falhou!");
    return;
  }

  // 6. Pega ponteiros para os tensores de entrada e saída
  tensor_entrada = interpretador->input(0);
  tensor_saida = interpretador->output(0);

  // 7. Checagens de segurança (verificando se o modelo é Float32)
  if (tensor_entrada->type != kTfLiteFloat32)
  {
    reporter_erro->Report("Tensor de entrada não é Float32!");
    return;
  }
  if (tensor_saida->type != kTfLiteFloat32)
  {
    reporter_erro->Report("Tensor de saída não é Float32!");
    return;
  }
  if (tensor_entrada->dims->data[1] != 3)
  {
    reporter_erro->Report("Input do modelo deve ter 3 características!");
    return;
  }

  // Se tudo deu certo:
  inicializado = true;
  Serial.println(F("Preditor de PA (TFLite) inicializado."));
}

// Destrutor
PreditorPressaoArterial_ESP32::~PreditorPressaoArterial_ESP32() {}
// Getter público para 'inicializado'
bool PreditorPressaoArterial_ESP32::IsInitialized() { return inicializado; }

// Implementação das funções auxiliares

std::vector<double> PreditorPressaoArterial_ESP32::gaussian_smooth(const std::vector<double> &segmento)
{
  if (segmento.size() < 3)
    return segmento;
  std::vector<double> suavizado(segmento.size());
  suavizado[0] = segmento[0];
  for (size_t i = 1; i < segmento.size() - 1; i++)
  {
    suavizado[i] = (segmento[i - 1] * 0.25) + (segmento[i] * 0.5) + (segmento[i + 1] * 0.25);
  }
  suavizado.back() = segmento.back();
  return suavizado;
}

double PreditorPressaoArterial_ESP32::trapz(const std::vector<double> &segmento)
{
  double area = 0.0;
  double dt = 1.0 / FREQ_AMOSTRAGEM;
  for (size_t i = 0; i < segmento.size() - 1; ++i)
  {
    area += (segmento[i] + segmento[i + 1]) * 0.5 * dt;
  }
  return area;
}

std::vector<int> PreditorPressaoArterial_ESP32::segmentacao_adaptativa_cpp(const std::vector<double> &sinal)
{
  if (sinal.size() < 3)
    return {};
  std::vector<double> suav(sinal.size());
  for (size_t i = 1; i < sinal.size() - 1; ++i)
  {
    suav[i] = (sinal[i - 1] + sinal[i] + sinal[i + 1]) / 3.0;
  }
  suav[0] = sinal[0];
  suav.back() = sinal.back();

  double r = *std::max_element(suav.begin(), suav.end()) - *std::min_element(suav.begin(), suav.end());
  if (r < 1e-6)
    return {};
  double h = std::accumulate(suav.begin(), suav.end(), 0.0) / suav.size() + 0.05 * r;

  int distancia_minima = floor((60.0 / 100.0) * FREQ_AMOSTRAGEM);

  std::vector<int> picos;
  for (size_t i = 1; i < suav.size() - 1; ++i)
  {
    if (suav[i] > suav[i - 1] && suav[i] > suav[i + 1] && suav[i] > h)
    {
      if (picos.empty() || (i - picos.back()) > distancia_minima)
      {
        picos.push_back(i);
      }
    }
  }
  return picos;
}

std::vector<int> PreditorPressaoArterial_ESP32::find_vales(const std::vector<double> &sinal, const std::vector<int> &picos)
{
  if (picos.size() < 2)
    return {};
  std::vector<int> vales;
  vales.push_back(0);
  for (size_t i = 0; i < picos.size() - 1; ++i)
  {
    int start = picos[i];
    int end = picos[i + 1];
    double min_val = sinal[start];
    int min_idx = start;
    for (int j = start + 1; j < end; j++)
    {
      if (sinal[j] < min_val)
      {
        min_val = sinal[j];
        min_idx = j;
      }
    }
    vales.push_back(min_idx);
  }
  vales.push_back(sinal.size() - 1);
  return vales;
}

Eigen::RowVector3d PreditorPressaoArterial_ESP32::extrair_3_features_cpp(const std::vector<double> &segmento, int inicio, int pico)
{
  Eigen::RowVector3d f;
  f.setZero();
  if (segmento.empty() || pico < inicio || pico >= (int)segmento.size())
    return f;

  double area = trapz(segmento);
  double sut = (pico - inicio) / FREQ_AMOSTRAGEM;

  std::vector<double> segmento_suavizado = gaussian_smooth(segmento);
  int idx_pico = pico - inicio;
  if (idx_pico < 0 || idx_pico >= segmento_suavizado.size())
    return f;
  double h_pulso = segmento_suavizado[idx_pico] - segmento_suavizado[0];
  if (h_pulso <= 1e-6)
    return f;
  double limiar = segmento_suavizado[0] + 0.25 * h_pulso;
  int i1 = -1, i2 = -1;
  for (int i = 0; i < (int)segmento_suavizado.size(); ++i)
  {
    if (segmento_suavizado[i] >= limiar)
    {
      if (i1 == -1)
        i1 = i;
      i2 = i;
    }
  }
  double largura = (i2 >= i1) ? (i2 - i1) / FREQ_AMOSTRAGEM : 0;

  f << area, sut, largura;
  return f;
}

// Função principal da IA: Recebe 10s de sinal (0-1) e retorna SBP/DBP
std::pair<float, float> PreditorPressaoArterial_ESP32::preverPressao(const std::vector<double> &segmento_bruto_ppg)
{
  if (segmento_bruto_ppg.size() < 800)
    return {0, 0};

  // 1. SEGMENTAÇÃO (Vale-a-Vale)
  std::vector<int> picos = segmentacao_adaptativa_cpp(segmento_bruto_ppg);
  std::vector<int> vales = find_vales(segmento_bruto_ppg, picos);

  Serial.printf("  [IA] Segmentação: %d picos, %d vales.\n", picos.size(), vales.size());

  if (picos.size() < 2 || vales.size() < 3)
  {
    reporter_erro->Report("Segmentação falhou.");
    return {0, 0};
  }

  // 2. LÓGICA DE PREDIÇÃO (Batimento-a-Batimento)
  std::vector<float> sbp_predicoes;
  std::vector<float> dbp_predicoes;
  sbp_predicoes.reserve(picos.size());
  dbp_predicoes.reserve(picos.size());
  int valid = 0;
  int vale_idx = 0;

  for (size_t i = 0; i < picos.size(); ++i)
  {
    int pico_atual = picos[i];

    while (vale_idx < vales.size() && vales[vale_idx] < pico_atual)
      vale_idx++;
    if (vale_idx == 0 || vale_idx >= vales.size() || vales[vale_idx] <= pico_atual)
      continue;
    int inicio_batimento = vales[vale_idx - 1];
    int fim_batimento = vales[vale_idx];

    if (inicio_batimento >= pico_atual || fim_batimento <= pico_atual || (fim_batimento - inicio_batimento) < 50)
      continue;

    std::vector<double> seg(segmento_bruto_ppg.begin() + inicio_batimento, segmento_bruto_ppg.begin() + fim_batimento + 1);

    Eigen::RowVector3d features = extrair_3_features_cpp(seg, 0, pico_atual - inicio_batimento);

    if (features.sum() < 1e-6)
      continue;
    valid++;

    Eigen::RowVector3d normalized = (features - media_features).array() / desvio_features.array();

    for (int j = 0; j < 3; ++j)
    {
      if (isnan(normalized[j]) || isinf(normalized[j]))
      {
        tensor_entrada->data.f[j] = 0.0f;
      }
      else
      {
        tensor_entrada->data.f[j] = (float)normalized[j];
      }
    }

    if (interpretador->Invoke() != kTfLiteOk)
    {
      reporter_erro->Report("Invoke falhou!");
      continue;
    }

    float sbp = tensor_saida->data.f[0];
    float dbp = tensor_saida->data.f[1];

    sbp_predicoes.push_back(sbp);
    dbp_predicoes.push_back(dbp);

  } // Fim do loop 'for' de batimentos

  Serial.printf("  [IA] Extração: %d batimentos válidos.\n", valid);

  if (valid == 0)
  {
    reporter_erro->Report("Nenhum batimento válido extraído.");
    return {0, 0};
  }

  // 3. CALCULA A MÉDIA
  float sbp_media = 0.0;
  float dbp_media = 0.0;
  for (float s : sbp_predicoes)
    sbp_media += s;
  for (float d : dbp_predicoes)
    dbp_media += d;

  sbp_media /= sbp_predicoes.size();
  dbp_media /= dbp_predicoes.size();

  Serial.printf("  [IA] Features (Média Bruta): (Calculo por batimento)\n");
  Serial.printf("  [IA] Features (Norm): (Calculo por batimento)\n");

  if (sbp_media == dbp_media)
  {
    reporter_erro->Report("  [IA] Erro: SBP e DBP idênticos.");
  }

  return {sbp_media, dbp_media};
}