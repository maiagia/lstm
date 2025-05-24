# Projeto de Previsão de Preços de Ações com LSTM

Este projeto implementa uma solução completa de Deep Learning para previsão de preços de fechamento de ações, utilizando redes neurais LSTM. O pipeline inclui coleta de dados, modelagem, criação de API RESTful com FastAPI e deploy com Docker.

## 📊 Visão Geral

> ℹ️ **Nota Importante:** Embora o modelo contenha um método `load_data()` com `yfinance`, ele foi adaptado para aceitar um `DataFrame` diretamente — geralmente fornecido por uma **API interna**. Isso garante que a predição ocorra com dados controlados e atualizados por chamadas autenticadas, em vez de depender da coleta ao vivo via `yfinance` durante a inferência.

A solução desenvolvida utiliza dados históricos de ações para treinar um modelo LSTM (Long Short Term Memory) capaz de prever os próximos valores de fechamento com base nos dados anteriores.

## 🧠 Tecnologias Utilizadas

- Python
- Pandas, NumPy, Scikit-learn
- TensorFlow / Keras
- FastAPI (API RESTful)
- Streamlit (Interface visual)
- Docker & Docker Compose

---

## ✅ Requisitos Atendidos

### 1. Coleta e Pré-processamento dos Dados
- Coleta inicial dos dados realizada com a biblioteca `yfinance` durante a fase de treinamento.
- Dados históricos são processados e normalizados com `MinMaxScaler`.
- Para **predições**, os dados **são enviados pelo cliente via API** (`POST /prever`) e não coletados diretamente pelo backend via yfinance.
- Também há rota para atualizar a base via API (`POST /carregar-dados`).

### 2. Desenvolvimento do Modelo LSTM
- Rede LSTM construída com Keras e TensorFlow.
- Treinamento com dados históricos com janela deslizante de 30 dias.
- Avaliação com métricas como MAE, MAPE, RMSE e SMAPE.
- Ajustes de hiperparâmetros para desempenho otimizado.

### 3. Salvamento e Exportação do Modelo
- Modelo salvo em formato `.h5` (`modelo_lstm.h5`).
- Scaler salvo como `.pkl` (`scaler.pkl`) para reuso no processo de normalização durante inferência.

### 4. Deploy do Modelo
- API RESTful criada com **FastAPI**.
- Endpoints:
  - `POST /prever`: recebe histórico via JSON e retorna previsão.
  - `POST /carregar-dados`: coleta novos dados do Yahoo Finance e atualiza base interna.
- Interface adicional desenvolvida com **Streamlit** para facilitar uso do modelo via navegador.

### 5. Escalabilidade e Monitoramento
- Deploy em múltiplos containers com Docker.
- `docker-compose.yml` define a arquitetura completa: API + Frontend.
- Logs e modularização da API facilitam futura instrumentação com Prometheus, Grafana, etc.

---

## 📁 Estrutura do Projeto

