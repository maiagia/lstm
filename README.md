# 📈 Previsão de Preços de Ações com LSTM

Este projeto é o desafio da Fase 4 do MBA em Machine Learning da FIAP. Ele tem como objetivo aplicar técnicas de Deep Learning — em especial, redes LSTM — para prever preços de ações da bolsa de valores com base em séries temporais. A aplicação final entrega uma API RESTful e uma interface interativa em Streamlit para interação com o modelo.

---

## 🚀 Visão Geral

- Coleta dados históricos da B3 via API customizada integrada com yFinance.
- Treina um modelo LSTM para prever o fechamento de ações.
- Disponibiliza endpoints RESTful para previsão e consulta de histórico.
- Cria uma interface visual com Streamlit.
- Utiliza Docker para facilitar o deploy e a escalabilidade da solução.

---

## ✅ Requisitos do Desafio e Como Foram Atendidos

### 1. Coleta e Pré-processamento dos Dados

- ✔️ Utilização da biblioteca `yfinance` para baixar dados históricos de ações diretamente na API, através do endpoint `/api/historico_preco`.
- ✔️ Interface Streamlit permite que o usuário defina a ação e o intervalo de datas.
- ✔️ Os dados são transformados e normalizados com `MinMaxScaler`.

### 2. Desenvolvimento do Modelo LSTM

- ✔️ Modelo LSTM construído com Keras e TensorFlow com duas camadas LSTM e camadas de Dropout.
- ✔️ O modelo é treinado com dados históricos passados pela API.
- ✔️ Métricas de avaliação implementadas: MAE, MAPE, RMSE e SMAPE.
- ✔️ Avaliação e predição para os próximos 7 dias são exibidas com gráfico e tabela.

### 3. Salvamento e Exportação do Modelo

- ✔️ Após o treinamento, o modelo é salvo no formato `.h5` e o scaler como `.pkl` usando `joblib`.
- ✔️ Isso permite reutilizar o modelo treinado para inferências futuras sem retraining.

### 4. Deploy do Modelo

- ✔️ API criada com FastAPI com endpoints:
  - `/api/historico_preco`: retorna dados históricos via yFinance.
  - `/api/predict`: recebe uma lista de preços e retorna a previsão.
- ✔️ Interface Streamlit comunica com a API para orquestrar toda a operação.
- ✔️ Geração de relatórios em PDF com tabela e gráfico de previsões.

### 5. Escalabilidade e Monitoramento

- ✔️ Deploy Dockerizado com dois containers: um para API (FastAPI) e outro para Frontend (Streamlit).
- ✔️ Comunicação entre serviços via `docker-compose` com rede interna isolada.

---

## 📁 Estrutura do Projeto

