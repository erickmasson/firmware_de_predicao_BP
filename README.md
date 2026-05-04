# Firmware de Predição BP

Este repositório contém o firmware desenvolvido para o projeto de predição de pressão arterial, utilizando a plataforma **PlatformIO**.

O modelo já está treinado para uso.

## Pré-requisitos

Antes de começar, você precisará ter instalado em sua máquina:
*   [Visual Studio Code (VS Code)](https://code.visualstudio.com/)
*   [Extension PlatformIO IDE](https://platformio.org/install/ide?install=vscode) (instalada dentro do VS Code)
*   [Git](https://git-scm.com/)

## Como instalar e abrir o projeto

Siga os passos abaixo para configurar o ambiente localmente:

### 1. Clonar o repositório
Abra o seu terminal e execute o comando:

```git clone https://github.com/erickmasson/firmware_de_predicao_BP.git```

### 2. Abrir no VS Code
1. Abra o **Visual Studio Code**.

2. No menu superior, vá em **File > Open Folder...** (ou Arquivo > Abrir Pasta).

3. Selecione a pasta ```firmware``` que você clonou.

### 3. Inicialização pelo PlatformIO
Assim que a pasta for aberta, o PlatformIO detectará automaticamente o arquivo platformio.ini.

* Aguarde alguns instantes para que ele baixe as bibliotecas e ferramentas necessárias (o progresso aparecerá na barra inferior).

* Quando o ícone da **formiga** (ícone do PlatformIO) aparecer na barra lateral esquerda, o projeto está pronto.
  
## Compilação e Upload
Use os ícones na barra de status inferior do VS Code:

* Check (✓): Compila o código (Build).

* Seta para a direita (→): Faz o upload para a placa (ESP32).

* Tomada/Plug: Abre o Monitor Serial para depuração.

## Estrutura do Projeto

* ```src/```: Contém o código fonte principal (```main.cpp```).
  
* ```lib/```: Bibliotecas privadas específicas do projeto.
  
* ```include/```: Arquivos de cabeçalho (.h).
  
* ```platformio.ini```: Configurações da placa, velocidade de serial e dependências.
