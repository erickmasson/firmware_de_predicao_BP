/*
 * main.ino (Arquivo Principal)
 *
 * 1) Inicia o Hardware (Sensor, Display e a rede neural)
 * 2) Lê o sensor MAX30102 a 400Hz
 * 3) Aplica os filtros (Passa-Alta, Média Móvel)
 * 4) Armazena o sinal filtrado em um buffer de 10s
 * 5) A cada 10s, normaliza o buffer e chama a IA para predição
 * 6) Exibe os resultados
 */

#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include "PreditorPressaoArterial_ESP32.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <deque>
#include <vector>
#include <algorithm>

// Objetos
MAX30105 sensorPPG;
PreditorPressaoArterial_ESP32 *preditorIA = nullptr;
#define LARGURA_TELA 128
#define ALTURA_TELA 64
Adafruit_SSD1306 display(LARGURA_TELA, ALTURA_TELA, &Wire, -1);

// Constantes
const int FREQ_AMOSTRAGEM = 400;
const int TAMANHO_BUFFER = FREQ_AMOSTRAGEM * 10;
unsigned long ultima_predicao = 0;
const unsigned long INTERVALO_PREDICAO = 10000;

// Buffers de sinal
std::deque<double> buffer_ppg;

// Variáveis do filtro Passa-Alta
const float ALFA_HP = 0.992;
float valor_bruto_anterior = 0.0;
float valor_hp_anterior = 0.0;
bool filtro_hp_iniciado = false;

// Variáveis do filtro de Média Móvel (suavização)
const int JANELA_SUAVIZACAO = 10;
float buffer_suavizacao[JANELA_SUAVIZACAO] = {0};
int indice_suavizacao = 0;
float soma_suavizacao = 0.0;
int contagem_suavizacao = 0;

// Buffer de média da saída (para estabilizar)
const int CONTAGEM_MEDIA = 3;
float historico_sbp[CONTAGEM_MEDIA] = {0};
float historico_dbp[CONTAGEM_MEDIA] = {0};
int indice_historico = 0;
bool historico_cheio = false;

// Offsets de calibração (opcional)
const float OFFSET_SBP = 0.0;
const float OFFSET_DBP = 0.0;

// Adiciona uma nova predição (SBP/DBP) ao buffer de média deslizante
void adicionar_ao_historico(float sbp, float dbp)
{
  historico_sbp[indice_historico] = sbp + OFFSET_SBP;
  historico_dbp[indice_historico] = dbp + OFFSET_DBP;
  indice_historico = (indice_historico + 1) % CONTAGEM_MEDIA;
  if (!historico_cheio && indice_historico == 0)
    historico_cheio = true;
}

// Calcula e retorna a média das últimas predições armazenadas
std::pair<float, float> calcular_media()
{
  if (!historico_cheio && indice_historico == 0)
    return {0, 0};
  int contagem = historico_cheio ? CONTAGEM_MEDIA : indice_historico;
  float soma_sbp = 0, soma_dbp = 0;
  for (int i = 0; i < contagem; i++)
  {
    soma_sbp += historico_sbp[i];
    soma_dbp += historico_dbp[i];
  }
  return {soma_sbp / contagem, soma_dbp / contagem};
}

// Função de inicialização principal
void setup()
{
  Serial.begin(115200);
  Serial.println("Iniciando Monitor de PA (3F - Area+SUT+Largura25)...");

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
  {
    Serial.println("OLED falhou");
    while (1)
      ;
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.println("Iniciando...");
  display.display();

  Wire.setClock(400000);

  Serial.println("Inicializando IA...");
  preditorIA = new PreditorPressaoArterial_ESP32();
  if (!preditorIA->IsInitialized())
  {
    Serial.println("ERRO: Modelo nao carregado.");
    while (1)
      ;
  }
  Serial.println("Modelo IA (3F) carregado!");

  Serial.println("Inicializando Sensor...");
  if (!sensorPPG.begin(Wire, I2C_SPEED_FAST))
  {
    Serial.println("MAX30102 nao encontrado.");
    ESP.restart();
  }
  sensorPPG.setup(10, 4, 2, 400, 411, 8192);
  sensorPPG.clearFIFO();
  Serial.println("Sensor OK.");

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Sistema Pronto");
  display.println("Aguardando contato com o sensor...");
  display.display();
}

// Loop principal
void loop()
{
  sensorPPG.check();

  while (sensorPPG.available())
  {
    long ir = sensorPPG.getIR();
    float raw = (float)ir;

    if (raw > 10000)
    {
      float ac = 0.0;

      if (!filtro_hp_iniciado)
      {
        valor_bruto_anterior = raw;
        valor_hp_anterior = 0.0;
        filtro_hp_iniciado = true;
      }
      else
      {
        ac = ALFA_HP * (valor_hp_anterior + raw - valor_bruto_anterior);
        valor_bruto_anterior = raw;
        valor_hp_anterior = ac;
      }

      soma_suavizacao -= buffer_suavizacao[indice_suavizacao];
      buffer_suavizacao[indice_suavizacao] = ac;
      soma_suavizacao += ac;
      indice_suavizacao = (indice_suavizacao + 1) % JANELA_SUAVIZACAO;
      if (contagem_suavizacao < JANELA_SUAVIZACAO)
        contagem_suavizacao++;

      if (contagem_suavizacao < JANELA_SUAVIZACAO)
      {
        sensorPPG.nextSample();
        continue;
      }

      float smoothed = soma_suavizacao / JANELA_SUAVIZACAO;
      float ppg_inv = -smoothed;

      buffer_ppg.push_back(ppg_inv);
      if (buffer_ppg.size() > TAMANHO_BUFFER)
        buffer_ppg.pop_front();

      unsigned long now = millis();
      if (now - ultima_predicao >= INTERVALO_PREDICAO && buffer_ppg.size() >= TAMANHO_BUFFER / 2)
      {
        ultima_predicao = now;

        Serial.printf("\n--- PREDIÇÃO (Amostras: %d) ---\n", buffer_ppg.size());

        std::vector<double> buffer_vec(buffer_ppg.begin(), buffer_ppg.end());

        auto minmax = std::minmax_element(buffer_vec.begin(), buffer_vec.end());
        double min_val = *minmax.first;
        double max_val = *minmax.second;
        double range = max_val - min_val;

        std::vector<double> buffer_normalizado;
        buffer_normalizado.reserve(buffer_vec.size());

        if (range < 1e-6)
        {
          Serial.println("Sinal plano detectado, pulando predição.");
          continue;
        }

        for (double val : buffer_vec)
        {
          buffer_normalizado.push_back((val - min_val) / range);
        }

        std::pair<float, float> pressoes = preditorIA->preverPressao(buffer_normalizado);
        float sbp_raw = pressoes.first;
        float dbp_raw = pressoes.second;

        if (sbp_raw > 60 && dbp_raw > 30 && sbp_raw > dbp_raw + 10 && sbp_raw < 200 && dbp_raw < 140)
        {
          adicionar_ao_historico(sbp_raw, dbp_raw);

          std::pair<float, float> media = calcular_media();
          float avg_sbp = media.first;
          float avg_dbp = media.second;

          if (historico_cheio || indice_historico > 0)
          {
            Serial.printf(">>> SBP: %.1f | DBP: %.1f mmHg (Media)\n", avg_sbp, avg_dbp);
            display.clearDisplay();
            display.setTextSize(2);
            display.setCursor(0, 0);
            display.printf("SBP:%3.0f\nDBP:%3.0f", avg_sbp, avg_dbp);
            display.setCursor(0, 50);
            display.setTextSize(1);
            display.printf("PP:%3.0f  Estavel", avg_sbp - avg_dbp);
            display.display();
          }
        }
        else
        {
          Serial.printf(">>> SBP: %.1f | DBP: %.1f -> Invalido\n", sbp_raw, dbp_raw);
          display.clearDisplay();
          display.setTextSize(1);
          display.setCursor(0, 0);
          display.println("Sinal instavel");
          display.println("Ajuste o sensor...");
          display.display();
        }
      }
    }
    else
    {
      filtro_hp_iniciado = false;
      contagem_suavizacao = 0;
      soma_suavizacao = 0;
      indice_suavizacao = 0;

      buffer_ppg.clear();
      indice_historico = 0;
      historico_cheio = false;

      display.clearDisplay();
      display.setCursor(0, 0);
      display.println("Aguardando contato com o sensor...");
      display.display();
    }
    sensorPPG.nextSample();
  }
}
