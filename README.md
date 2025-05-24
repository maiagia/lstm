# Projeto de Previsão de Preços de Ações com LSTM

Este projeto implementa uma solução completa de Deep Learning para previsão de preços de fechamento de ações, utilizando redes neurais LSTM. O pipeline completo inclui coleta de dados, pré-processamento, modelagem, avaliação, salvamento do modelo, criação de API RESTful com FastAPI, e preparação para deploy via Docker.

## 📊 Visão Geral

A solução desenvolvida utiliza dados históricos de ações para treinar um modelo LSTM (Long Short Term Memory) capaz de prever os próximos valores de fechamento com base nos dados anteriores.

## 🧠 Tecnologias Utilizadas

- Python
- yfinance
- Pandas, NumPy, Scikit-learn
- TensorFlow / Keras
- FastAPI
- Streamlit (Interface Web)
- Docker & Docker Compose

---

## ✅ Requisitos Atendidos

### 1. Coleta e Pré-processamento dos Dados
- Utilização da biblioteca `yfinance` para obter dados históricos de ações (por exemplo, da Disney - símbolo `DIS`).
- Definição de janela deslizante para features (ex: 60 dias).
- Normalização dos dados com `MinMaxScaler`.
- Divisão de dados em treino e teste.

### 2. Desenvolvimento do Modelo LSTM
- Implementação de rede LSTM com camadas LSTM e Dense usando Keras.
- Treinamento com ajuste de hiperparâmetros (épocas, batch size).
- Avaliação do modelo utilizando métricas como MAE, MSE, RMSE.
- Salvamento do modelo (`modelo_lstm.h5`) e scaler (`scaler.pkl`).

### 3. Salvamento e Exportação do Modelo
- O modelo treinado é salvo no formato `.h5`.
- O scaler de normalização é salvo como `.pkl`.

### 4. Deploy do Modelo
- Criação de uma API RESTful com FastAPI para realizar previsões.
- Rotas:
  - `POST /prever`: recebe histórico e retorna previsão.
  - `POST /carregar-dados`: busca novos dados do Yahoo Finance.
- Organização modular da API com rotas, modelos, serviços e utilitários.

### 5. Escalabilidade e Monitoramento
- Preparação para deploy em contêiner Docker.
- Dockerfile e docker-compose incluídos.
- Separação de containers para a API e para a interface com Streamlit.
- Logs e estrutura modular facilitam futura integração com ferramentas de monitoramento.

---

## 📁 Estrutura do Projeto


