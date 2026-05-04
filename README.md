# Firmware de Predição BP

Este repositório contém o firmware para o monitor de pressão arterial baseado em ESP32, utilizando **TensorFlow Lite for Microcontrollers**. O sistema processa sinais de PPG (sensor MAX30102) a 400Hz e realiza a predição de pressão sistólica e diastólica em tempo real.

> **Status do Modelo:** O projeto já inclui um modelo pré-treinado e funcional. Se desejar entender como o treinamento foi realizado ou treinar com seus próprios dados, acesse o repositório de treinamento: **[Pipeline de IA - BP Prediction](https://github.com/erickmasson/treinamento_IA_BP)**.

---

## Pontos de Integração (IA -> Firmware)

Se você optar por re-treinar o modelo ou utilizar novos dados, existem três pontos cruciais no código que devem ser atualizados. Estes valores são gerados ao final do pipeline de treinamento:

### 1. Pesos da Rede Neural (`model_data.h`)
O arquivo `include/model_data.h` contém o array hexadecimal que representa o modelo. 
No arquivo `src/PreditorPressaoArterial_ESP32.cpp`, o interpretador carrega o modelo através do nome da array:
```cpp
modelo = tflite::GetModel(ann_MLP_3F_tflite);
```
*Caso mude o nome da array no arquivo `.h` ao usar o comando `xxd`, este campo deve ser atualizado.*

### 2. Parâmetros de Normalização (`PreditorPressaoArterial_ESP32.cpp`)
A rede neural exige que os dados de entrada estejam na mesma escala usada no treinamento (Z-Score). Os valores de **Média** e **Desvio Padrão** gerados na **Etapa 2** do treinamento devem ser inseridos no construtor da classe:
```cpp
// Valores exatos gerados pelo script Python (Etapa 2)
media_features << 0.28579714, 0.20255669, 0.38727512;
desvio_features << 0.0787428, 0.07602341, 0.12433794;
```

---

## Pré-requisitos

*   [Visual Studio Code (VS Code)](https://code.visualstudio.com/)
*   [Extension PlatformIO IDE](https://platformio.org/install/ide?install=vscode)
*   [Git](https://git-scm.com/)

---

## Como instalar e abrir o projeto

1.  **Clonar o repositório:**
    ```
    git clone [https://github.com/erickmasson/firmware_de_predicao_BP.git](https://github.com/erickmasson/firmware_de_predicao_BP.git)
    ```
2.  **Abrir no VS Code:** Vá em `File > Open Folder...` e selecione a pasta do projeto.
3.  **Inicialização:** O PlatformIO baixará as dependências automaticamente (TensorFlowLite_ESP32, Eigen, etc.).

---

## Compilação e Upload

Use os atalhos na barra inferior do VS Code:
*   **Check (✓):** Compila o código (Build).
*   **Seta (→):** Upload para a placa (ESP32).
*   **Plug:** Monitor Serial (115200 baud).

---

## Estrutura do Projeto

*   `src/`: Lógica principal (`main.ino`) e implementação da IA (`.cpp`).
*   `include/`: Arquivos de cabeçalho e o modelo binário (`model_data.h`).
*   `lib/`: Bibliotecas locais (ex: Eigen).
*   `platformio.ini`: Configurações de hardware e dependências.

---

## Observação Técnica
O firmware está configurado para uma **Frequência de Amostragem de 400Hz**. Qualquer alteração na taxa de leitura do sensor exigirá um novo treinamento do modelo para manter a precisão das características extraídas.
